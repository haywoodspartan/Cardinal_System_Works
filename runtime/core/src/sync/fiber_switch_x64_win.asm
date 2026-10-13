; =============================================================================
; Cardinal - x86-64 Windows context switch (MASM, ml64).
;
;   void cardinal_fiber_switch(void** save_sp /* rcx */, void* load_sp /* rdx */)
;
; Saves the Windows x64 callee-saved state on the CURRENT stack, stores the
; resulting rsp into *save_sp, loads rsp from load_sp and restores the same
; state from there, then `ret` resumes whoever last saved that frame.
;
; Frame layout, offsets from the saved rsp (0x110 bytes incl. return address):
;   000..09F  xmm6 .. xmm15 (16 B each)
;   0A0       mxcsr            0A4  x87 control word
;   0A8       NT_TIB.StackBase 0B0  NT_TIB.StackLimit   0B8  TEB.DeallocationStack
;   0C0       pad
;   0C8 r15   0D0 r14   0D8 r13   0E0 r12   0E8 rdi   0F0 rsi   0F8 rbx   100 rbp
;   108       return address
;
; A fresh context (fiber.cpp make_context) is such a frame with r12 = entry,
; r13 = arg and the return address = cardinal_fiber_trampoline, which calls
; entry(arg) with the ABI's shadow space.
;
; The TIB stack bounds are swapped with the stack so __chkstk probes, SEH /
; C++ unwinding and minidumps see the fiber stack as the thread's stack.
; =============================================================================

.code

PUBLIC cardinal_fiber_switch
cardinal_fiber_switch PROC
    push rbp
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 0C8h                       ; 8 pushes + 0C8h keeps rsp 16-aligned

    movaps xmmword ptr [rsp+000h], xmm6
    movaps xmmword ptr [rsp+010h], xmm7
    movaps xmmword ptr [rsp+020h], xmm8
    movaps xmmword ptr [rsp+030h], xmm9
    movaps xmmword ptr [rsp+040h], xmm10
    movaps xmmword ptr [rsp+050h], xmm11
    movaps xmmword ptr [rsp+060h], xmm12
    movaps xmmword ptr [rsp+070h], xmm13
    movaps xmmword ptr [rsp+080h], xmm14
    movaps xmmword ptr [rsp+090h], xmm15
    stmxcsr dword ptr [rsp+0A0h]
    fnstcw  word ptr  [rsp+0A4h]

    mov  r10, qword ptr gs:[030h]        ; NT_TIB* (TEB)
    mov  rax, qword ptr [r10+008h]       ; StackBase
    mov  qword ptr [rsp+0A8h], rax
    mov  rax, qword ptr [r10+010h]       ; StackLimit
    mov  qword ptr [rsp+0B0h], rax
    mov  rax, qword ptr [r10+01478h]     ; DeallocationStack
    mov  qword ptr [rsp+0B8h], rax

    mov  qword ptr [rcx], rsp            ; *save_sp = this frame
    mov  rsp, rdx                        ; switch stacks

    mov  r10, qword ptr gs:[030h]
    mov  rax, qword ptr [rsp+0A8h]
    mov  qword ptr [r10+008h], rax
    mov  rax, qword ptr [rsp+0B0h]
    mov  qword ptr [r10+010h], rax
    mov  rax, qword ptr [rsp+0B8h]
    mov  qword ptr [r10+01478h], rax

    ldmxcsr dword ptr [rsp+0A0h]
    fldcw   word ptr  [rsp+0A4h]
    movaps xmm6,  xmmword ptr [rsp+000h]
    movaps xmm7,  xmmword ptr [rsp+010h]
    movaps xmm8,  xmmword ptr [rsp+020h]
    movaps xmm9,  xmmword ptr [rsp+030h]
    movaps xmm10, xmmword ptr [rsp+040h]
    movaps xmm11, xmmword ptr [rsp+050h]
    movaps xmm12, xmmword ptr [rsp+060h]
    movaps xmm13, xmmword ptr [rsp+070h]
    movaps xmm14, xmmword ptr [rsp+080h]
    movaps xmm15, xmmword ptr [rsp+090h]

    add  rsp, 0C8h
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rdi
    pop  rsi
    pop  rbx
    pop  rbp
    ret
cardinal_fiber_switch ENDP

; Entered by the `ret` of the first switch into a fresh context: rsp is
; 16-aligned here (one slot above the frame), r12 = entry, r13 = arg.
PUBLIC cardinal_fiber_trampoline
cardinal_fiber_trampoline PROC
    sub  rsp, 20h                        ; shadow space, keeps 16-byte alignment
    mov  rcx, r13
    call r12                             ; entry(arg) - never returns
    ud2
cardinal_fiber_trampoline ENDP

END
