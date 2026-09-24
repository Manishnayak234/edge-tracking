#pragma once

namespace edge_tracking {

// YUV encoding of a frame: which matrix (BT.601 / BT.709) and which value range
// (limited: Y in 16..235, full: 0..255). Needed to convert YUV to RGB correctly.
enum class ColorSpace {
    Unknown,
    Bt601Limited,
    Bt601Full,
    Bt709Limited,
    Bt709Full,
};

inline const char* to_string(ColorSpace cs) {
    switch (cs) {
        case ColorSpace::Bt601Limited: return "BT.601 limited";
        case ColorSpace::Bt601Full: return "BT.601 full";
        case ColorSpace::Bt709Limited: return "BT.709 limited";
        case ColorSpace::Bt709Full: return "BT.709 full";
        case ColorSpace::Unknown: break;
    }
    return "unknown";
}

}  // namespace edge_tracking
