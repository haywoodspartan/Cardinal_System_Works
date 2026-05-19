// =============================================================================
// Cardinal — Networking: UDP transport (Winsock).
//
// Non-blocking UDP socket, hand-rolled framing + a tiny connection
// handshake (no external deps — same ethos as the hand-rolled physics
// / audio). Authoritative client-server: a server listen()s and accepts
// many peers; a client connect()s to one server.
//
// Channels: Unreliable is fire-and-forget. ReliableOrdered is a real
// cumulative-ack reliable channel — per-peer outbound sequence + an
// unacked list retransmitted on poll() (the per-frame pump doubles as
// the reliability timer), receiver Acks every reliable datagram, and
// delivers strictly in order with a reorder buffer for early arrivals
// + dedupe of retransmits. Simple (per-packet ack, not a bitfield) but
// correct under loss/reorder/duplication.
//
// set_conditions() injects artificial inbound packet loss (hand-rolled
// xorshift, no <random>). NOTE the deliberate difference from the
// loopback sim: the loopback has no reliability machinery, so it must
// exempt ReliableOrdered (perturbing it there is just unrecoverable
// breakage). UDP has a real ack+retransmit layer, so loss here is
// modelled the realistic way — the link drops datagrams blind to
// channel/type (data AND acks alike), and the reliability layer is
// expected to recover ReliableOrdered regardless while Unreliable
// simply degrades. (latency/jitter aren't modelled on UDP — a real
// socket already has real RTT; the loopback owns deterministic
// time-based perturbation.)
//
// Single TU, both paths behind one CARDINAL_PLATFORM_WINDOWS switch —
// mirrors audio_wasapi.cpp. CMake links ws2_32 on Windows.
// =============================================================================

#include <cardinal/net/net.hpp>
#include <cardinal/core/log.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#ifndef WIN32_LEAN_AND_MEAN          // already defined project-wide
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include <cardinal/core/chrono.hpp>       // cardinal::chrono
#include <cardinal/core/cstring.hpp>      // cardinal::memcpy
#include <cardinal/core/containers.hpp>   // cardinal::deque/vector
#include <cardinal/core/utility.hpp>      // cardinal::move/pair

namespace cardinal::net {

namespace {

constexpr u32 kProtocolId = 0x434E5431;   // 'CNT1' — reject stray datagrams

// Per-peer cap on early-arrival (future-seq) datagrams held for
// reordering. seq is attacker-controlled off the wire, so an unbounded
// buffer is a trivial remote memory-exhaustion DoS (flood distinct high
// seqs that never let in_expected advance). A real reorder window is a
// handful of packets; 256 (~512 KB/peer worst case at MTU) is generous.
// ReliableOrdered means the sender retransmits, so dropping a future
// datagram past the window is correct, not lossy.
constexpr usize kMaxReorderBuffered = 256;

enum class PktType : u8 {
    Connect = 0, ConnectAck = 1, Data = 2, Disconnect = 3, Ack = 4
};

// ReliableOrdered tuning. Cumulative per-packet ack (simple + correct,
// chattier than a bitfield — fine for the current scale).
constexpr double kRetransmitInterval = 0.10;   // seconds before resend

#pragma pack(push, 1)
struct PktHeader {
    u32 protocol;     // kProtocolId
    u8  type;         // PktType
    u8  channel;      // Channel (Data only)
    u16 pad;
    u32 seq;          // reliable sequence (0 for unreliable / control)
};
#pragma pack(pop)
static_assert(sizeof(PktHeader) == 12, "PktHeader must be tightly packed");

double now_seconds() {
    using namespace cardinal::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

bool same_addr(const sockaddr_in& a, const sockaddr_in& b) {
    return a.sin_addr.s_addr == b.sin_addr.s_addr && a.sin_port == b.sin_port;
}

class UdpTransport final : public Transport {
public:
    UdpTransport() {
        WSADATA wsa{};
        wsa_ok_ = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    ~UdpTransport() override {
        if (sock_ != INVALID_SOCKET) closesocket(sock_);
        if (wsa_ok_) WSACleanup();
    }

    bool listen(u16 port) override {
        if (!open_socket_()) return false;
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port);
        if (bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr))
                == SOCKET_ERROR) {
            cardinal::log::warnf("net/udp", "bind(%u) failed (%d)",
                port, WSAGetLastError());
            return false;
        }
        is_server_ = true;
        cardinal::log::infof("net/udp", "listening on udp/%u", port);
        return true;
    }

    bool connect(const char* host, u16 port) override {
        if (!open_socket_()) return false;
        // Ephemeral local bind so recvfrom works.
        sockaddr_in local{};
        local.sin_family      = AF_INET;
        local.sin_addr.s_addr = INADDR_ANY;
        local.sin_port        = 0;
        bind(sock_, reinterpret_cast<sockaddr*>(&local), sizeof(local));

        server_addr_ = {};
        server_addr_.sin_family = AF_INET;
        server_addr_.sin_port   = htons(port);
        if (inet_pton(AF_INET, host ? host : "127.0.0.1",
                      &server_addr_.sin_addr) != 1) {
            cardinal::log::warnf("net/udp", "bad host '%s'", host ? host : "");
            return false;
        }
        is_server_ = false;
        // Fire the handshake; the server's ConnectAck (seen in poll())
        // promotes us to connected + raises the Connected event.
        PktHeader h = make_hdr_(PktType::Connect, Channel::ReliableOrdered, 0);
        sendto(sock_, reinterpret_cast<const char*>(&h), sizeof(h), 0,
               reinterpret_cast<sockaddr*>(&server_addr_),
               sizeof(server_addr_));
        return true;
    }

    void send(PeerId to, Channel ch, const void* data, u32 size) override {
        if (sock_ == INVALID_SOCKET) return;
        if (Peer* p = peer_by_id_(to)) send_to_(*p, ch, data, size);
    }

    void broadcast(Channel ch, const void* data, u32 size) override {
        if (sock_ == INVALID_SOCKET) return;
        for (auto& p : peers_) send_to_(p, ch, data, size);
    }

    usize poll(cardinal::vector<NetEvent>& out) override {
        if (sock_ == INVALID_SOCKET) return 0;
        usize n = 0;
        char buf[2048];
        for (;;) {
            sockaddr_in from{};
            int fromlen = sizeof(from);
            const int got = recvfrom(sock_, buf, sizeof(buf), 0,
                reinterpret_cast<sockaddr*>(&from), &fromlen);
            if (got == SOCKET_ERROR) break;          // WOULDBLOCK: drained
            if (got < static_cast<int>(sizeof(PktHeader))) continue;
            PktHeader h{};
            cardinal::memcpy(&h, buf, sizeof(h));
            if (h.protocol != kProtocolId) continue;  // not ours

            // Injected link loss: drop the datagram as if it never
            // arrived — blind to type, so Data AND Acks can vanish.
            // ReliableOrdered must still get through (retransmit);
            // Unreliable just degrades.
            if (cond_active_ && rng_unit_() < cond_.loss) continue;

            switch (static_cast<PktType>(h.type)) {
            case PktType::Connect: {
                if (!is_server_) break;
                PeerId pid = find_peer_(from);
                if (pid == kInvalidPeer) pid = add_peer_(from);
                PktHeader ack = make_hdr_(PktType::ConnectAck,
                                          Channel::ReliableOrdered, 0);
                sendto(sock_, reinterpret_cast<const char*>(&ack),
                       sizeof(ack), 0,
                       reinterpret_cast<sockaddr*>(&from), fromlen);
                out.push_back(evt_(NetEventKind::Connected, pid,
                                   Channel::ReliableOrdered, nullptr, 0));
                ++n;
                break;
            }
            case PktType::ConnectAck: {
                if (is_server_ || connected_) break;
                connected_ = true;
                const PeerId pid = add_peer_(from);   // the server
                out.push_back(evt_(NetEventKind::Connected, pid,
                                   Channel::ReliableOrdered, nullptr, 0));
                ++n;
                break;
            }
            case PktType::Data: {
                Peer* pr = peer_by_addr_(from);
                if (pr == nullptr) break;              // unknown sender
                const u32     plen = static_cast<u32>(got) - sizeof(PktHeader);
                const char*   pay  = buf + sizeof(PktHeader);
                const Channel ch   = static_cast<Channel>(h.channel);

                if (ch != Channel::ReliableOrdered) {
                    out.push_back(evt_(NetEventKind::Message, pr->id,
                                       ch, pay, plen));
                    ++n;
                    break;
                }
                // ReliableOrdered: always Ack (even duplicates — the
                // sender may not have received the earlier Ack), then
                // deliver in order, buffering early arrivals.
                send_ack_(from, h.seq);
                if (h.seq < pr->in_expected) break;    // already delivered
                if (h.seq == pr->in_expected) {
                    out.push_back(evt_(NetEventKind::Message, pr->id,
                                       ch, pay, plen));
                    ++n;
                    ++pr->in_expected;
                    // Drain any now-contiguous buffered seqs.
                    bool progressed = true;
                    while (progressed) {
                        progressed = false;
                        for (auto it = pr->reorder.begin();
                             it != pr->reorder.end(); ++it) {
                            if (it->first != pr->in_expected) continue;
                            out.push_back(evt_(NetEventKind::Message,
                                pr->id, Channel::ReliableOrdered,
                                it->second.data(),
                                static_cast<u32>(it->second.size())));
                            ++n;
                            ++pr->in_expected;
                            pr->reorder.erase(it);
                            progressed = true;
                            break;
                        }
                    }
                } else {
                    // Future seq — buffer once (dedupe retransmits).
                    bool have = false;
                    for (auto& e : pr->reorder)
                        if (e.first == h.seq) { have = true; break; }
                    // Bounded reorder window — see kMaxReorderBuffered.
                    // Past the cap, drop: ReliableOrdered retransmits, so
                    // a peer >256 ahead of in_expected (lossy or hostile)
                    // can't grow this buffer without limit.
                    if (!have && pr->reorder.size() < kMaxReorderBuffered) {
                        pr->reorder.emplace_back(h.seq,
                            cardinal::vector<char>(pay, pay + plen));
                    }
                }
                break;
            }
            case PktType::Ack: {
                Peer* pr = peer_by_addr_(from);
                if (pr == nullptr) break;
                for (auto it = pr->unacked.begin();
                     it != pr->unacked.end(); ++it) {
                    if (it->seq == h.seq) { pr->unacked.erase(it); break; }
                }
                break;
            }
            case PktType::Disconnect: {
                const PeerId pid = find_peer_(from);
                if (pid == kInvalidPeer) break;
                out.push_back(evt_(NetEventKind::Disconnected, pid,
                                   Channel::ReliableOrdered, nullptr, 0));
                remove_peer_(pid);
                ++n;
                break;
            }
            }
        }

        // Retransmit pass — any reliable datagram unacked for longer
        // than kRetransmitInterval goes back out. poll() is the per-
        // frame pump so this doubles as the reliability timer.
        const double now = now_seconds();
        for (auto& p : peers_) {
            for (auto& op : p.unacked) {
                if (now - op.last_send < kRetransmitInterval) continue;
                raw_send_(p.addr, op.bytes.data(),
                          static_cast<int>(op.bytes.size()));
                op.last_send = now;
            }
        }
        return n;
    }

    void disconnect(PeerId peer) override {
        Peer* pr = peer_by_id_(peer);
        if (pr == nullptr) return;
        const sockaddr_in* a = &pr->addr;
        PktHeader h = make_hdr_(PktType::Disconnect,
                                Channel::ReliableOrdered, 0);
        sendto(sock_, reinterpret_cast<const char*>(&h), sizeof(h), 0,
               reinterpret_cast<const sockaddr*>(a), sizeof(*a));
        remove_peer_(peer);
    }

    bool  is_server()  const noexcept override { return is_server_; }
    usize peer_count() const noexcept override { return peers_.size(); }

    void set_conditions(const NetConditions& c) noexcept override {
        cond_        = c;
        cond_active_ = (c.loss > 0.0f);
        if (c.seed != seeded_) {           // reseed only on change
            rng_    = c.seed ? c.seed : 0x9E3779B9u;
            seeded_ = c.seed;
        }
    }

private:
    struct OutPkt {
        u32               seq{0};
        cardinal::vector<char> bytes;     // full datagram (header + payload)
        double            last_send{0.0};
    };
    struct Peer {
        sockaddr_in addr{};
        PeerId      id{kInvalidPeer};
        // ReliableOrdered outbound: monotonically-assigned seq + the
        // not-yet-acked datagrams (retransmitted on poll()).
        u32                 out_seq{0};
        cardinal::vector<OutPkt> unacked;
        // ReliableOrdered inbound: next seq to deliver + a reorder
        // buffer for early arrivals. Seqs start at 1 (0 = unreliable).
        u32 in_expected{1};
        cardinal::vector<cardinal::pair<u32, cardinal::vector<char>>> reorder;
    };

    bool open_socket_() {
        if (!wsa_ok_) return false;
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == INVALID_SOCKET) return false;
        u_long nb = 1;
        ioctlsocket(sock_, FIONBIO, &nb);             // non-blocking
        return true;
    }

    PktHeader make_hdr_(PktType t, Channel ch, u32 seq) const {
        PktHeader h{};
        h.protocol = kProtocolId;
        h.type     = static_cast<u8>(t);
        h.channel  = static_cast<u8>(ch);
        h.seq      = seq;
        return h;
    }

    void raw_send_(const sockaddr_in& dst, const char* bytes, int len) {
        sendto(sock_, bytes, len, 0,
               reinterpret_cast<const sockaddr*>(&dst), sizeof(dst));
    }

    void send_to_(Peer& peer, Channel ch, const void* data, u32 size) {
        const u32 seq = (ch == Channel::ReliableOrdered)
                      ? ++peer.out_seq : 0u;
        cardinal::vector<char> pkt(sizeof(PktHeader) + size);
        PktHeader h = make_hdr_(PktType::Data, ch, seq);
        cardinal::memcpy(pkt.data(), &h, sizeof(h));
        if (data != nullptr && size > 0)
            cardinal::memcpy(pkt.data() + sizeof(h), data, size);
        raw_send_(peer.addr, pkt.data(), static_cast<int>(pkt.size()));
        // Reliable: retain until Ack'd; retransmitted on poll().
        if (ch == Channel::ReliableOrdered) {
            OutPkt op;
            op.seq       = seq;
            op.bytes     = cardinal::move(pkt);
            op.last_send = now_seconds();
            peer.unacked.push_back(cardinal::move(op));
        }
    }

    void send_ack_(const sockaddr_in& to, u32 seq) {
        PktHeader a = make_hdr_(PktType::Ack, Channel::ReliableOrdered, seq);
        raw_send_(to, reinterpret_cast<const char*>(&a), sizeof(a));
    }

    Peer* peer_by_addr_(const sockaddr_in& a) {
        for (auto& p : peers_) if (same_addr(p.addr, a)) return &p;
        return nullptr;
    }
    Peer* peer_by_id_(PeerId id) {
        for (auto& p : peers_) if (p.id == id) return &p;
        return nullptr;
    }
    PeerId find_peer_(const sockaddr_in& a) const {
        for (const auto& p : peers_) if (same_addr(p.addr, a)) return p.id;
        return kInvalidPeer;
    }
    PeerId add_peer_(const sockaddr_in& a) {
        const PeerId id = next_peer_++;
        peers_.push_back(Peer{a, id});
        return id;
    }
    void remove_peer_(PeerId id) {
        for (auto it = peers_.begin(); it != peers_.end(); ++it)
            if (it->id == id) { peers_.erase(it); return; }
    }

    // Hand-rolled xorshift32 for injected loss — no <random>, same
    // no-deps ethos as the loopback sim. [0,1) draw.
    float rng_unit_() noexcept {
        u32 x = rng_;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        rng_ = x;
        return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
    }

    static NetEvent evt_(NetEventKind k, PeerId p, Channel ch,
                         const char* data, u32 size) {
        NetEvent e;
        e.kind = k; e.peer = p; e.channel = ch;
        if (data != nullptr && size > 0) {
            e.data.resize(size);
            cardinal::memcpy(e.data.data(), data, size);
        }
        return e;
    }

    SOCKET            sock_{INVALID_SOCKET};
    bool              wsa_ok_{false};
    bool              is_server_{false};
    bool              connected_{false};
    sockaddr_in       server_addr_{};
    cardinal::vector<Peer> peers_;
    PeerId            next_peer_{1};
    // Injected link conditions (loss only on UDP — see file header).
    NetConditions     cond_{};
    bool              cond_active_{false};
    u32               rng_{0x9E3779B9u};
    u32               seeded_{0};
};

}  // namespace

cardinal::unique_ptr<Transport> Transport::create_udp() {
    return cardinal::make_unique<UdpTransport>();
}

}  // namespace cardinal::net

#else   // !CARDINAL_PLATFORM_WINDOWS

namespace cardinal::net {

cardinal::unique_ptr<Transport> Transport::create_udp() {
    cardinal::log::infof("net",
        "UDP transport: no socket backend on this platform yet "
        "— use create_loopback()");
    return nullptr;
}

}  // namespace cardinal::net

#endif  // CARDINAL_PLATFORM_WINDOWS
