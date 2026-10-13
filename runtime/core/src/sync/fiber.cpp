#include <cardinal/core/sync/fiber.hpp>

#if CARDINAL_FIBER_ASM

#include <cstdint>
#include <cstring>

#if CARDINAL_PLATFORM_WINDOWS
    #include <Windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// The two assembly routines (fiber_switch_x64_win.asm / fiber_switch_x64_sysv.S).
extern "C" {
void cardinal_fiber_switch(void** save_sp, void* load_sp) noexcept;
void cardinal_fiber_trampoline() noexcept;   // only its address is used: a fresh
                                             // context "returns" into it.
}

namespace cardinal::fiber {

namespace {

usize page_size() noexcept {
#if CARDINAL_PLATFORM_WINDOWS
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return si.dwPageSize ? static_cast<usize>(si.dwPageSize) : 4096u;
#else
    const long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? static_cast<usize>(p) : 4096u;
#endif
}

inline usize round_up(usize v, usize a) noexcept { return (v + a - 1) / a * a; }

// Frame layout written by the assembly switch — offsets from the saved rsp.
// Must match the .asm / .S files byte for byte.
#if CARDINAL_PLATFORM_WINDOWS
constexpr usize kFrameBytes     = 0x110;   // 10 xmm + control words + TIB + 8 GPRs + return address
constexpr usize kOffMxcsr       = 0xA0;
constexpr usize kOffFpuCw       = 0xA4;
constexpr usize kOffStackBase   = 0xA8;
constexpr usize kOffStackLimit  = 0xB0;
constexpr usize kOffDealloc     = 0xB8;
constexpr usize kOffR13         = 0xD8;    // entry argument
constexpr usize kOffR12         = 0xE0;    // entry function
constexpr usize kOffReturn      = 0x108;
constexpr u16   kDefaultFpuCw   = 0x027F;  // MSVC x64 default x87 control word
#else
constexpr usize kFrameBytes     = 0x40;    // control words + 6 GPRs + return address
constexpr usize kOffMxcsr       = 0x00;
constexpr usize kOffFpuCw       = 0x04;
constexpr usize kOffR13         = 0x18;
constexpr usize kOffR12         = 0x20;
constexpr usize kOffReturn      = 0x38;
constexpr u16   kDefaultFpuCw   = 0x037F;  // SysV default x87 control word
#endif
constexpr u32 kDefaultMxcsr = 0x1F80;      // all exceptions masked, round-to-nearest

template <class T>
inline void put(u8* frame, usize off, T v) noexcept { std::memcpy(frame + off, &v, sizeof v); }

}  // namespace

Stack allocate_stack(usize usable_bytes) noexcept {
    Stack s{};
    const usize page   = page_size();
    const usize usable = round_up(usable_bytes == 0 ? page : usable_bytes, page);
    const usize total  = usable + page;   // + guard page at the bottom
#if CARDINAL_PLATFORM_WINDOWS
    void* mem = VirtualAlloc(nullptr, total, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (mem == nullptr) return s;
    DWORD old = 0;
    if (!VirtualProtect(mem, page, PAGE_NOACCESS, &old)) {
        VirtualFree(mem, 0, MEM_RELEASE);
        return s;
    }
#else
    void* mem = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return s;
    if (mprotect(mem, page, PROT_NONE) != 0) {
        munmap(mem, total);
        return s;
    }
#endif
    s.alloc      = mem;
    s.alloc_size = total;
    s.base       = static_cast<u8*>(mem) + page;
    s.size       = usable;
    return s;
}

void free_stack(Stack& s) noexcept {
    if (s.alloc != nullptr) {
#if CARDINAL_PLATFORM_WINDOWS
        VirtualFree(s.alloc, 0, MEM_RELEASE);
#else
        munmap(s.alloc, s.alloc_size);
#endif
    }
    s = Stack{};
}

void make_context(Context& ctx, const Stack& stack, EntryFn entry, void* arg) noexcept {
    // Top of the usable region, 16-byte aligned. The first switch into this
    // frame pops the return address and lands in the trampoline with
    // rsp == top, which is the alignment the ABI wants at a call site.
    u8* top = static_cast<u8*>(stack.base) + stack.size;
    top = reinterpret_cast<u8*>(reinterpret_cast<std::uintptr_t>(top) & ~static_cast<std::uintptr_t>(15));
    u8* frame = top - kFrameBytes;
    std::memset(frame, 0, kFrameBytes);
    put<u32>(frame, kOffMxcsr, kDefaultMxcsr);
    put<u16>(frame, kOffFpuCw, kDefaultFpuCw);
#if CARDINAL_PLATFORM_WINDOWS
    put<void*>(frame, kOffStackBase,  top);
    put<void*>(frame, kOffStackLimit, stack.base);
    put<void*>(frame, kOffDealloc,    stack.alloc);
#endif
    put<void*>(frame, kOffR13,    arg);
    put<void*>(frame, kOffR12,    reinterpret_cast<void*>(entry));
    put<void*>(frame, kOffReturn, reinterpret_cast<void*>(&cardinal_fiber_trampoline));
    ctx.sp = frame;
}

void switch_context(Context& from, Context& to) noexcept {
    cardinal_fiber_switch(&from.sp, to.sp);
}

}  // namespace cardinal::fiber

#endif  // CARDINAL_FIBER_ASM
