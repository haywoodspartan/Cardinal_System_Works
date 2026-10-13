#pragma once

// =============================================================================
// Cardinal — user-level context switch (the job system's fibers).
//
// A hand-rolled x86-64 context switch replaces Win32 fibers / ucontext. The
// assembly routine saves exactly the state the ABI says a call must preserve:
//   Windows x64 : RBX RBP RSI RDI R12-R15, XMM6-XMM15, MXCSR, x87 control word,
//                 and the three NT_TIB stack-bound fields (StackBase,
//                 StackLimit, DeallocationStack) so stack probes, SEH / C++
//                 unwinding and the crash + watchdog minidumps all see the
//                 fiber's stack as a real stack;
//   SysV x86-64 : RBX RBP R12-R15, MXCSR, x87 control word.
//
// Because the switch is an ordinary (opaque) call to the compiler, every live
// value is either in a callee-saved register the routine preserves or in
// memory the compiler must reload after the call — which is what lets
// jobs.cpp build at /O2 again. SwitchToFiber had the same contract on paper;
// whatever broke under /O2 was never pinned down, and this path is small
// enough to audit end to end. ucontext on Linux additionally cost a
// rt_sigprocmask syscall per switch.
//
// Stacks are VirtualAlloc'd / mmap'd with a no-access guard page at the
// bottom: an overflow faults immediately instead of corrupting a neighbour.
//
// x86-64 only (CARDINAL_FIBER_ASM); other targets keep the OS primitives.
// =============================================================================

#include <cardinal/core/platform.hpp>
#include <cardinal/core/types.hpp>

#if CARDINAL_ARCH_X64 && (CARDINAL_PLATFORM_WINDOWS || CARDINAL_PLATFORM_LINUX)
    #define CARDINAL_FIBER_ASM 1
#else
    #define CARDINAL_FIBER_ASM 0
#endif

namespace cardinal::fiber {

// A suspended context: the stack pointer of its saved register frame. A
// thread's native stack needs no preparation — its first use as the `from`
// side of switch_context() records it.
struct Context {
    void* sp{nullptr};
};

// A fiber stack. `base` is the lowest USABLE byte (the guard page sits just
// below it) and `size` the usable byte count; alloc / alloc_size cover the
// whole mapping for release.
struct Stack {
    void* base{nullptr};
    usize size{0};
    void* alloc{nullptr};
    usize alloc_size{0};
};

// Entry point of a fresh context. It must never return: there is no frame
// to return to on a fiber stack (the trampoline traps if it does).
using EntryFn = void (*)(void* arg) noexcept;

// Allocates `usable_bytes` (rounded up to whole pages) plus one guard page.
// base == nullptr on failure.
Stack allocate_stack(usize usable_bytes) noexcept;
void  free_stack(Stack& s) noexcept;

// Prepares `ctx` so the first switch_context() into it runs entry(arg) on
// `stack`, with the ABI's alignment, shadow space and default floating-point
// control words.
void make_context(Context& ctx, const Stack& stack, EntryFn entry, void* arg) noexcept;

// Saves the calling context into `from` and resumes `to`. Returns when some
// other context switches back into `from`.
void switch_context(Context& from, Context& to) noexcept;

}  // namespace cardinal::fiber
