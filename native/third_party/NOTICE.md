# Third-party notices

The native tracker (`native/src/track/`) is MotionMan's own C++ code. It implements published algorithms whose
reference implementations are under the permissive licences below. No source file was copied; the structure of
the assignment solver follows lap's, so its notice is kept here as the licence asks.

## ByteTrack (algorithm)
Yifu Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
Reference implementation: https://github.com/FoundationVision/ByteTrack, MIT License,
Copyright (c) 2021 Yifu Zhang.

## lap (Jonker-Volgenant assignment)
`native/src/track/lapjv.cpp` follows the phase structure and scan order of lap's dense `lapjv`
(https://github.com/gatagat/lap, the Python package the reference tracker calls) so that equal-cost ties
resolve the same way. BSD 2-Clause License, Copyright (c) 2012-2017, Tomas Kazmar.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided
    that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
       following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
       following disclaimer in the documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
    OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.

## Behavioural reference (not copied)
Ultralytics 8.4.196 `BYTETracker` (AGPL-3.0) is the Python tracker the legacy backend runs. The native tracker
is checked against its outputs by replaying detections (`tools/tracker_replay/`); none of its code is used here.
