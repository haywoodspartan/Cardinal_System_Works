# =============================================================================
# Cardinal - x86-64 SysV context switch (GAS, Intel syntax).
#
#   void cardinal_fiber_switch(void** save_sp /* rdi */, void* load_sp /* rsi */)
#
# Frame layout, offsets from the saved rsp (0x40 bytes incl. return address):
#   00 mxcsr   04 x87 control word
#   08 r15   10 r14   18 r13   20 r12   28 rbx   30 rbp
#   38 return address
#
# A fresh context (fiber.cpp make_context) is such a frame with r12 = entry,
# r13 = arg and the return address = cardinal_fiber_trampoline.
# =============================================================================

    .text
    .intel_syntax noprefix

    .globl cardinal_fiber_switch
    .type  cardinal_fiber_switch, @function
cardinal_fiber_switch:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 8                          # 6 pushes + 8 keeps rsp 16-aligned
    stmxcsr dword ptr [rsp]
    fnstcw  word ptr  [rsp+4]

    mov  qword ptr [rdi], rsp            # *save_sp = this frame
    mov  rsp, rsi                        # switch stacks

    ldmxcsr dword ptr [rsp]
    fldcw   word ptr  [rsp+4]
    add  rsp, 8
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
    .size cardinal_fiber_switch, .-cardinal_fiber_switch

# Entered by the `ret` of the first switch into a fresh context: rsp is
# 16-aligned here, r12 = entry, r13 = arg.
    .globl cardinal_fiber_trampoline
    .type  cardinal_fiber_trampoline, @function
cardinal_fiber_trampoline:
    mov  rdi, r13
    call r12                             # entry(arg) - never returns
    ud2
    .size cardinal_fiber_trampoline, .-cardinal_fiber_trampoline

    .section .note.GNU-stack,"",@progbits
