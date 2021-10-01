from __future__ import annotations

#from ctypes import *
#import os
#import subprocess
#import numpy as np
#import scipy.stats as st
#import os

import os
import sys
_HERE_ = os.path.dirname(os.path.abspath(__file__))
print(sys.path[0])
sys.path.append(sys.path[0]+"/..")
sys.path.append(sys.path[0]+"/.")
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
from .gemmini import *
from .harness_gemmini import ENV, GemmTestBuilder
import pytest


# --------------------------------------------------------------------------- #
#   MatMul Demo
# --------------------------------------------------------------------------- #

def test_simple_perfect():
    T = GemmTestBuilder('simple_perfect')
    # T.add_body(['gemm_init_mem();',
    #             'gemm_acc_init_mem();',
    #             'gemmini_flush(0);',
    #             ''])
    T.add_body(['gemmini_flush(0);',
                ''])
    T.add_body(["simple_perfect_lib_Context *ctxt;"])

    NN = 16
    MM = 16
    KK = 16

    T.alloc_dram_2i8('x', NN, KK, '1')
    T.alloc_dram_2i8('y', KK, MM, '1')
    T.alloc_dram_f32('a_scale', '1.0f')
    T.alloc_dram_f32('b_scale', '1.0f')
    T.alloc_dram_f32('c_scale', '1.0f')
    T.alloc_dram_2i8('z_cpu', NN, MM, '0') # expected result
    T.alloc_dram_2i8('z_gemmini', NN, MM, '0')

    @proc
    def simple_perfect(
      N : size,
      M : size,
      K : size,
      a_scale : f32,
      b_scale : f32,
      c_scale : f32,
      acc     : bool,
      A : i8[N,K] @ DRAM,
      B : i8[K,M] @ DRAM,
      C : i8[N,M] @ DRAM,
    ):
        assert N == 16
        assert M == 16
        assert K == 16

        for i in par(0,16):
            for j in par(0,16):
                res : i32 @ DRAM
                res = 0.0
                for k in par(0,16):
                    tmp_a : f32
                    tmp_a = A[i,k]
                    tmp_a = tmp_a * a_scale
                    a : i8 @ DRAM
                    a = tmp_a

                    tmp_b : f32
                    tmp_b = B[k,j]
                    tmp_b = tmp_b * b_scale
                    b : i8 @ DRAM
                    b = tmp_b

                    a2 : i32
                    b2 : i32
                    a2 = a
                    b2 = b
                    res += a2*b2

                tmp_res : i8
                if acc == True:
                    tmp_res = relu(res)
                else:
                    tmp_res = res

                tmp_res2 : f32
                tmp_res2 = tmp_res
                tmp_res2 = tmp_res2 * c_scale
                clamp(tmp_res2, tmp_res)
                C[i,j] = tmp_res

    simple_perfect = simple_perfect.lift_alloc('res : _', n_lifts=2)
    # simple_perfect = simple_perfect.reorder('i', 'j')

    T.add_proc(simple_perfect)

    T.start_timer('gemmini')
    T.add_body([f'simple_perfect(ctxt, {NN}, {MM}, {KK}, a_scale, b_scale, c_scale, false, x, y, z_cpu);',
                f'gemmini_fence();'])
    T.stop_timer('gemmini', 'Cycles for GEMMINI version')

    T.compile().run()

    print(simple_perfect)

    simple_perfect.check_effects()
