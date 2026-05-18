#include <cardinal/cine/cine.hpp>

#include <cardinal/core/log.hpp>

#include <algorithm>

namespace cardinal::cine {

void Player::play()  noexcept {
    if (seq_ != nullptr) seq_->playing = true;
}
void Player::pause() noexcept {
    if (seq_ != nullptr) seq_->playing = false;
}
void Player::stop()  noexcept {
    if (seq_ != nullptr) {
        seq_->playing      = false;
        seq_->play_head_s  = 0.0f;
        prev_time_s_       = 0.0f;
    }
}
void Player::seek(float t) {
    if (seq_ == nullptr) return;
    seq_->play_head_s = std::clamp(t, 0.0f, seq_->duration_s);
    prev_time_s_      = seq_->play_head_s;
}

void Player::tick(float dt) {
    if (seq_ == nullptr || !seq_->playing) return;
    const float prev = seq_->play_head_s;
    float t = prev + dt * seq_->speed;
    if (t >= seq_->duration_s) {
        if (seq_->looping) {
            // Wrap; on wrap, fire any events between [prev, duration] then
            // continue from 0 in this same tick. Cheap recursion: split.
            const float carry = t - seq_->duration_s;
            seq_->play_head_s = seq_->duration_s;
            // Apply tracks for the tail of this lap.
            for (auto& tv : seq_->tracks) {
                std::visit([&](auto& tr) {
                    using TT = std::decay_t<decltype(tr)>;
                    if constexpr (std::is_same_v<TT, AnimationTrack>) {
                        if (tr.player) tr.player->tick(0.0f);
                    } else if constexpr (std::is_same_v<TT, CameraTrack>) {
                        for (const auto& sh : tr.shots)
                            if (sh.time_s > prev && sh.time_s <= seq_->duration_s)
                                if (on_camera_) on_camera_(sh.camera_actor_id, sh.blend_seconds);
                    } else if constexpr (std::is_same_v<TT, AudioTrack>) {
                        for (const auto& ev : tr.events)
                            if (ev.time_s > prev && ev.time_s <= seq_->duration_s) {
                                if (audio_) {
                                    if (ev.is_3d) audio_->play_3d(ev.cue_id, ev.position, ev.channel, ev.volume);
                                    else          audio_->play_2d(ev.cue_id, ev.channel, ev.volume);
                                }
                            }
                    } else if constexpr (std::is_same_v<TT, EventTrack>) {
                        for (const auto& ev : tr.events)
                            if (ev.time_s > prev && ev.time_s <= seq_->duration_s)
                                if (on_event_) on_event_(ev.event_name, ev.payload);
                    }
                }, tv);
            }
            // Reset, continue from carry.
            seq_->play_head_s = 0.0f;
            prev_time_s_      = 0.0f;
            t = carry;
        } else {
            t = seq_->duration_s;
            seq_->playing = false;
        }
    }

    // Apply tracks for [prev, t].
    for (auto& tv : seq_->tracks) {
        std::visit([&](auto& tr) {
            using TT = std::decay_t<decltype(tr)>;
            if constexpr (std::is_same_v<TT, AnimationTrack>) {
                if (tr.player) {
                    const float local_t = t - tr.start_time_s;
                    if (local_t >= 0.0f && local_t <= tr.length_s) {
                        tr.player->seek(local_t);
                        tr.player->tick(0.0f);   // apply current pose
                    }
                }
            } else if constexpr (std::is_same_v<TT, CameraTrack>) {
                for (const auto& sh : tr.shots)
                    if (sh.time_s > prev_time_s_ && sh.time_s <= t)
                        if (on_camera_) on_camera_(sh.camera_actor_id, sh.blend_seconds);
            } else if constexpr (std::is_same_v<TT, AudioTrack>) {
                for (const auto& ev : tr.events)
                    if (ev.time_s > prev_time_s_ && ev.time_s <= t) {
                        if (audio_) {
                            if (ev.is_3d) audio_->play_3d(ev.cue_id, ev.position, ev.channel, ev.volume);
                            else          audio_->play_2d(ev.cue_id, ev.channel, ev.volume);
                        }
                    }
            } else if constexpr (std::is_same_v<TT, EventTrack>) {
                for (const auto& ev : tr.events)
                    if (ev.time_s > prev_time_s_ && ev.time_s <= t)
                        if (on_event_) on_event_(ev.event_name, ev.payload);
            }
        }, tv);
    }

    seq_->play_head_s = t;
    prev_time_s_      = t;
}

}  // namespace cardinal::cine
