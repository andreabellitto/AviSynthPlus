// Avisynth v2.5.  Copyright 2007 Ben Rudiak-Gould et al.
// http://www.avisynth.org

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.


#include <avisynth.h>
#include "internal.h"
#include <intrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>

int AviHelper_ImageSize(const VideoInfo *vi, bool AVIPadScanlines, bool v210, bool v410, bool r210, bool R10k) {
  int image_size;
  if (vi->pixel_type == VideoInfo::CS_YUV444P16 || vi->pixel_type == VideoInfo::CS_YUVA444P16)
    { // Y416 packed 4444 U,Y,V,A
    image_size = vi->width * vi->height * 4 * sizeof(uint16_t);
  }
  else if (vi->pixel_type == VideoInfo::CS_RGBP10 && r210)
  { // 3x10bit packed RGB, 64 aligned
    image_size = ((vi->width + 63) / 64) * 256 * vi->height; // 4 byte/pixel: 32bits for 3x10 bits
  }
  else if (vi->pixel_type == VideoInfo::CS_RGBP10 && R10k)
  { // 3x10bit packed RGB, no aligment
    image_size = vi->width * 4 * vi->height; // 4 byte/pixel: 32bits for 3x10 bits
  }
  else if (vi->pixel_type == VideoInfo::CS_YUV444P10 && v410)
  { // v410 packed 444 U,Y,V
    image_size = vi->width * vi->height * 4; // 4 byte/pixel: 32bits for 3x10 bits
  }
  else if (vi->pixel_type == VideoInfo::CS_YUV422P10 && v210)
  {
    image_size = ((16 * ((vi->width + 5) / 6) + 127) & ~127);
    image_size *= vi->height;
  }
  else if ((vi->IsRGB() && !vi->IsPlanar()) || vi->IsYUY2() || vi->IsY() || AVIPadScanlines) {
    // incl. all packed RGBs
    image_size = vi->BMPSize();
  }
  else { // Packed size
    if (vi->IsPlanar() && vi->IsRGB()) {
      image_size = (vi->RowSize(PLANAR_G) * vi->height);
      if (vi->IsPlanarRGBA()) // not supported yet, but for the sake of completeness
        image_size *= 4;
      else
        image_size *= 3;
    }
    else {
      image_size = vi->RowSize(PLANAR_U);
      if (image_size) {
        image_size *= vi->height;
        image_size >>= vi->GetPlaneHeightSubsampling(PLANAR_U);
        image_size *= 2;
      }
      image_size += vi->RowSize(PLANAR_Y) * vi->height;
    }
  }
  return image_size;
}

template<bool hasAlpha>
void ToY416_sse2(uint8_t *outbuf, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height)
{
  const int wmod4 = (width / 4) * 4;

  // UYVA
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wmod4; x += 4) {
      // read 4x4 pixels, store 2x(4x2) pixels
      auto u = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(uptr + x * sizeof(uint16_t)));
      auto y = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(yptr + x * sizeof(uint16_t)));
      auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(vptr + x * sizeof(uint16_t)));
      __m128i a;
      if (hasAlpha)
        a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(aptr + x * sizeof(uint16_t)));
      else
        a = _mm_set1_epi16(-1); // transparent alpha 0xFFFF
      auto uy = _mm_unpacklo_epi16(u, y);
      auto va = _mm_unpacklo_epi16(v, a);
      auto uyva_lo = _mm_unpacklo_epi32(uy, va);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(outbuf + 4 * sizeof(uint16_t) * x), uyva_lo);
      auto uyva_hi = _mm_unpackhi_epi32(uy, va);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(outbuf + 16 + 4 * sizeof(uint16_t) * x), uyva_hi);
    }

    for (int x = wmod4; x < width; x++) {
      reinterpret_cast<uint16_t *>(outbuf)[x * 4 + 0] = reinterpret_cast<const uint16_t *>(uptr)[x];
      reinterpret_cast<uint16_t *>(outbuf)[x * 4 + 1] = reinterpret_cast<const uint16_t *>(yptr)[x];
      reinterpret_cast<uint16_t *>(outbuf)[x * 4 + 2] = reinterpret_cast<const uint16_t *>(vptr)[x];
      reinterpret_cast<uint16_t *>(outbuf)[x * 4 + 3] = hasAlpha ? reinterpret_cast<const uint16_t *>(aptr)[x] : 0xFFFF;
    }
    outbuf += out_pitch;
    yptr += ypitch;
    uptr += uvpitch;
    vptr += uvpitch;
    aptr += apitch;
  }
}

// instantiate
template void ToY416_sse2<false>(uint8_t *outbuf, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height);
template void ToY416_sse2<true>(uint8_t *outbuf, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height);

template<bool hasAlpha>
void ToY416_c(uint8_t *outbuf8, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height)
{
  uint16_t *outbuf = reinterpret_cast<uint16_t *>(outbuf8);
  out_pitch /= sizeof(uint16_t);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      outbuf[x * 4 + 0] = reinterpret_cast<const uint16_t *>(uptr)[x];
      outbuf[x * 4 + 1] = reinterpret_cast<const uint16_t *>(yptr)[x];
      outbuf[x * 4 + 2] = reinterpret_cast<const uint16_t *>(vptr)[x];
      outbuf[x * 4 + 3] = hasAlpha ? reinterpret_cast<const uint16_t *>(aptr)[x] : 0xFFFF;
    }
    outbuf += out_pitch;
    yptr += ypitch;
    uptr += uvpitch;
    vptr += uvpitch;
    aptr += apitch;
  }
}
// instantiate
template void ToY416_c<false>(uint8_t *outbuf, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height);
template void ToY416_c<true>(uint8_t *outbuf, int out_pitch, const uint8_t *yptr, int ypitch, const uint8_t *uptr, const uint8_t *vptr, int uvpitch, const uint8_t *aptr, int apitch, int width, int height);

template<bool hasAlpha>
void FromY416_c(uint8_t *yptr, int ypitch, uint8_t *uptr, uint8_t *vptr, int uvpitch, uint8_t *aptr, int apitch, 
  const uint8_t *srcp8, int srcpitch,
  int width, int height)
{
  const uint16_t *srcp = reinterpret_cast<const uint16_t *>(srcp8);
  srcpitch /= sizeof(uint16_t);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      reinterpret_cast<uint16_t *>(uptr)[x] = srcp[x * 4 + 0];
      reinterpret_cast<uint16_t *>(yptr)[x] = srcp[x * 4 + 1];
      reinterpret_cast<uint16_t *>(vptr)[x] = srcp[x * 4 + 2];
      if(hasAlpha)
        reinterpret_cast<uint16_t *>(aptr)[x] = srcp[x * 4 + 3];
    }
    srcp += srcpitch;
    yptr += ypitch;
    uptr += uvpitch;
    vptr += uvpitch;
    aptr += apitch;
  }
}
// instantiate
template void FromY416_c<false>(uint8_t *yptr, int ypitch, uint8_t *uptr, uint8_t *vptr, int uvpitch, uint8_t *aptr, int apitch, const uint8_t *srcp8, int srcpitch, int width, int height);
template void FromY416_c<true>(uint8_t *yptr, int ypitch, uint8_t *uptr, uint8_t *vptr, int uvpitch, uint8_t *aptr, int apitch, const uint8_t *srcp8, int srcpitch, int width, int height);

// Helpers for 10 bit RGB -> Planar RGB

static __forceinline uint32_t swap32(uint32_t x) {
  x = (x & 0x0000FFFFu) << 16 | (x & 0xFFFF0000u) >> 16;
  x = (x & 0x00FF00FFu) << 8 | (x & 0xFF00FF00u) >> 8;
  return x;
}

void From_r210_c(uint8_t *rptr, uint8_t *gptr, uint8_t *bptr, int pitch, uint8_t *srcp8, int srcpitch, int width, int height)
{
  // XXrrrrrr rrrrgggg ggggggbb bbbbbbbb
  // BigEndian
  const uint32_t *srcp = reinterpret_cast<const uint32_t *>(srcp8);
  srcpitch /= sizeof(uint32_t);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = swap32(srcp[x]);
      reinterpret_cast<uint16_t *>(bptr)[x] = (rgb >> 0) & 0x3FF;
      reinterpret_cast<uint16_t *>(gptr)[x] = (rgb >> 10) & 0x3FF;
      reinterpret_cast<uint16_t *>(rptr)[x] = (rgb >> 20) & 0x3FF;
    }
    srcp += srcpitch;
    gptr += pitch;
    rptr += pitch;
    bptr += pitch;
  }
}

void From_R10k_c(uint8_t *rptr, uint8_t *gptr, uint8_t *bptr, int pitch, uint8_t *srcp8, int srcpitch, int width, int height)
{
  // rrrrrrrr rrgggggg ggggbbbb bbbbbbxx
  // BigEndian
  const uint32_t *srcp = reinterpret_cast<const uint32_t *>(srcp8);
  srcpitch /= sizeof(uint32_t);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const uint32_t rgb = swap32(srcp[x]);
      reinterpret_cast<uint16_t *>(bptr)[x] = (rgb >> 2) & 0x3FF;
      reinterpret_cast<uint16_t *>(gptr)[x] = (rgb >> 12) & 0x3FF;
      reinterpret_cast<uint16_t *>(rptr)[x] = (rgb >> 22) & 0x3FF;
    }
    srcp += srcpitch;
    gptr += pitch;
    rptr += pitch;
    bptr += pitch;
  }
}

// Helpers for b64a <-> RGB64

static __forceinline uint64_t swap64(uint64_t x) {
  x = (x & 0x00000000FFFFFFFFULL) << 32 | (x & 0xFFFFFFFF00000000ULL) >> 32;
  x = (x & 0x0000FFFF0000FFFFULL) << 16 | (x & 0xFFFF0000FFFF0000ULL) >> 16;
  x = (x & 0x00FF00FF00FF00FFULL) << 8 | (x & 0xFF00FF00FF00FF00ULL) >> 8;
  return x;
}

template<bool hasSSSE3>
static __forceinline __m128i _mm_bswap_epi64(__m128i x)
{
  // Reverse order of bytes in each 64-bit word.
  if (hasSSSE3) {
    return _mm_shuffle_epi8(x, _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
  }
  else {
    // Swap bytes in each 16-bit word:
    __m128i a = _mm_or_si128(
      _mm_slli_epi16(x, 8),
      _mm_srli_epi16(x, 8));

    // Reverse all 16-bit words in 64-bit halves:
    a = _mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
    a = _mm_shufflehi_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));

    return a;
  }
}

static __forceinline uint16_t swap16(uint16_t x) {
  return (x & 0x00FF) << 8 | (x & 0xFF00) >> 8;
}

// 3x1
void bgr_to_rgbBE_c(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height)
{
  // todo sse2
  // R G B R G B R G
  // B R G B R G B R
  // G B R G B R G B

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint16_t r = swap16(reinterpret_cast<const uint16_t *>(src)[x * 3 + 0]);
      uint16_t g = swap16(reinterpret_cast<const uint16_t *>(src)[x * 3 + 1]);
      uint16_t b = swap16(reinterpret_cast<const uint16_t *>(src)[x * 3 + 2]);
      reinterpret_cast<uint16_t*>(pdst)[x * 3 + 0] = b;
      reinterpret_cast<uint16_t*>(pdst)[x * 3 + 1] = g;
      reinterpret_cast<uint16_t*>(pdst)[x * 3 + 2] = r;
    }
    src += srcpitch;
    pdst += dstpitch;
  }
}

// 4x16: two-way symmetric
template<bool hasSSSE3>
void bgra_to_argbBE_sse(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height)
{
  const int wmod2 = (width / 2) * 2; // 2x64bit
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wmod2; x += 2) {
      __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 8 * x));
      a = _mm_bswap_epi64<hasSSSE3>(a);
      _mm_store_si128(reinterpret_cast<__m128i *>(pdst + 8 * x), a);
    }
    if (wmod2 < width) {
      __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(src + 8 * wmod2));
      a = _mm_bswap_epi64<hasSSSE3>(a);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(pdst + 8 * wmod2), a);
    }
    src += srcpitch;
    pdst += dstpitch;
  }
}

//instantiate
template void bgra_to_argbBE_sse<false>(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height);
template void bgra_to_argbBE_sse<true>(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height);

// 4x16: two-way symmetric
void bgra_to_argbBE_c(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint64_t a = reinterpret_cast<const uint64_t *>(src)[x]; // bgra -> argb+byte swap
      a = swap64(a);
      reinterpret_cast<uint64_t*>(pdst)[x] = a;
    }
    src += srcpitch;
    pdst += dstpitch;
  }
}

// Helpers for YUV420(422)P10<->P010 and YUV420(422)P16<->P016 conversion

template<bool before>
static void prepare_luma_shift6_c(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (before)
        reinterpret_cast<uint16_t*>(pdst)[x] = reinterpret_cast<const uint16_t *>(src)[x] << 6;
      else
        reinterpret_cast<uint16_t*>(pdst)[x] = reinterpret_cast<const uint16_t *>(src)[x] >> 6;
    }
    src += srcpitch;
    pdst += dstpitch;
  }
}

template<bool before>
static void prepare_luma_shift6_sse2(uint8_t* pdst, int dstpitch, const uint8_t *src, int srcpitch, int width, int height)
{
  const int modw = (width / 8) * 8; // 8 uv pairs at a time
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < modw; x += 8) {
      __m128i y = _mm_load_si128(reinterpret_cast<const __m128i *>(reinterpret_cast<const uint16_t *>(src) + x));
      if (before)
        y = _mm_slli_epi16(y, 6); // make 10->16 bits
      else
        y = _mm_srli_epi16(y, 6); // make 16->10 bits
      _mm_store_si128(reinterpret_cast<__m128i *>(reinterpret_cast<uint16_t *>(pdst) + x), y);
    }

    for (int x = modw; x < width; x++) {
      if (before)
        reinterpret_cast<uint16_t*>(pdst)[x] = reinterpret_cast<const uint16_t *>(src)[x] << 6;
      else
        reinterpret_cast<uint16_t*>(pdst)[x] = reinterpret_cast<const uint16_t *>(src)[x] >> 6;
    }
    src += srcpitch;
    pdst += dstpitch;
  }
}

template<bool shift6>
static void prepare_to_interleaved_uv_c(uint8_t* pdst, int dstpitch, const uint8_t *srcu, const uint8_t *srcv, int pitchUV, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint16_t u, v;
      if (shift6) {
        u = reinterpret_cast<const uint16_t *>(srcu)[x] << 6; // make 10->16 bits
        v = reinterpret_cast<const uint16_t *>(srcv)[x] << 6; // make 10->16 bits
      }
      else {
        u = reinterpret_cast<const uint16_t *>(srcu)[x];
        v = reinterpret_cast<const uint16_t *>(srcv)[x];
      }
      uint32_t uv = (v << 16) | u;
      reinterpret_cast<uint32_t*>(pdst)[x] = uv;
    }
    srcu += pitchUV;
    srcv += pitchUV;
    pdst += dstpitch;
  }
}

template<bool shift6>
static void prepare_to_interleaved_uv_sse2(uint8_t* pdst, int dstpitch, const uint8_t *srcu, const uint8_t *srcv, int pitchUV, int width, int height)
{
  const int modw = (width / 8) * 8; // 8 uv pairs at a time
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < modw; x += 8) {
      __m128i u = _mm_load_si128(reinterpret_cast<const __m128i *>(reinterpret_cast<const uint16_t *>(srcu) + x));
      __m128i v = _mm_load_si128(reinterpret_cast<const __m128i *>(reinterpret_cast<const uint16_t *>(srcv) + x));
      if (shift6) {
        u = _mm_slli_epi16(u, 6); // make 10->16 bits
        v = _mm_slli_epi16(v, 6);
      }
      __m128i uv;
      uv = _mm_unpacklo_epi16(u, v); // (v << 16) | u;
      _mm_store_si128(reinterpret_cast<__m128i *>(reinterpret_cast<uint32_t *>(pdst) + x), uv);
      uv = _mm_unpackhi_epi16(u, v); // (v << 16) | u;
      _mm_store_si128(reinterpret_cast<__m128i *>(reinterpret_cast<uint32_t *>(pdst) + x + 4), uv);
    }

    for (int x = modw; x < width; x++) {
      uint16_t u, v;
      if (shift6) {
        u = reinterpret_cast<const uint16_t *>(srcu)[x] << 6; // make 10->16 bits
        v = reinterpret_cast<const uint16_t *>(srcv)[x] << 6; // make 10->16 bits
      }
      else {
        u = reinterpret_cast<const uint16_t *>(srcu)[x];
        v = reinterpret_cast<const uint16_t *>(srcv)[x];
      }
      uint32_t uv = (v << 16) | u;
      reinterpret_cast<uint32_t*>(pdst)[x] = uv;
    }
    srcu += pitchUV;
    srcv += pitchUV;
    pdst += dstpitch;
  }
}

template<bool shift6>
static void prepare_from_interleaved_uv_c(uint8_t* pdstu, uint8_t* pdstv, int pitchUV, const uint8_t *src, int srcpitch, int width, int height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      uint32_t uv = reinterpret_cast<const uint32_t*>(src)[x];
      uint16_t u = uv & 0xFFFF;
      uint16_t v = uv >> 16;
      if (shift6) {
        u >>= 6;
        v >>= 6;
      }
      reinterpret_cast<uint16_t*>(pdstu)[x] = u;
      reinterpret_cast<uint16_t*>(pdstv)[x] = v;
    }
    pdstu += pitchUV;
    pdstv += pitchUV;
    src += srcpitch;
  }
}

template<bool shift6, bool hasSSE41>
static void prepare_from_interleaved_uv_sse2(uint8_t* pdstu, uint8_t* pdstv, int pitchUV, const uint8_t *src, int srcpitch, int width, int height)
{
  const int modw = (width / 8) * 8;
  auto mask0000FFFF = _mm_set1_epi32(0x0000FFFF);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < modw; x += 8) {
      auto uv_lo = _mm_load_si128(reinterpret_cast<const __m128i *>(reinterpret_cast<const uint32_t*>(src) + x));
      auto uv_hi = _mm_load_si128(reinterpret_cast<const __m128i *>(reinterpret_cast<const uint32_t*>(src) + x + 4));
      if (shift6) {
        uv_lo = _mm_srli_epi16(uv_lo, 6);
        uv_hi = _mm_srli_epi16(uv_hi, 6);
      }
      auto u_lo = _mm_and_si128(uv_lo, mask0000FFFF);
      auto u_hi = _mm_and_si128(uv_hi, mask0000FFFF);
      auto u = shift6 ? _mm_packs_epi32(u_lo, u_hi) : (hasSSE41 ? _mm_packus_epi32(u_lo, u_hi) : _MM_PACKUS_EPI32(u_lo, u_hi));
      _mm_store_si128(reinterpret_cast<__m128i *>(reinterpret_cast<uint16_t*>(pdstu) + x), u);

      auto v_lo = _mm_srli_epi32(uv_lo, 16);
      auto v_hi = _mm_srli_epi32(uv_hi, 16);
      auto v = shift6 ? _mm_packs_epi32(v_lo, v_hi) : (hasSSE41 ? _mm_packus_epi32(v_lo, v_hi) : _MM_PACKUS_EPI32(v_lo, v_hi));
      _mm_store_si128(reinterpret_cast<__m128i *>(reinterpret_cast<uint16_t*>(pdstv) + x), v);
    }

    for (int x = modw; x < width; x++) {
      uint32_t uv = reinterpret_cast<const uint32_t*>(src)[x];
      uint16_t u = uv & 0xFFFF;
      uint16_t v = uv >> 16;
      if (shift6) {
        u >>= 6;
        v >>= 6;
      }
      reinterpret_cast<uint16_t*>(pdstu)[x] = u;
      reinterpret_cast<uint16_t*>(pdstv)[x] = v;
    }

    pdstu += pitchUV;
    pdstv += pitchUV;
    src += srcpitch;
  }
}


void yuv422p10_to_v210(BYTE *dstp, const BYTE *srcp_y, int srcpitch, const BYTE *srcp_u, const BYTE *srcp_v, int srcpitch_uv, int width, int height)
{
  int ppitch_y = srcpitch / sizeof(uint16_t);
  int ppitch_uv = srcpitch_uv / sizeof(uint16_t);
  const uint16_t *yptr = (const uint16_t *)srcp_y;
  const uint16_t *uptr = (const uint16_t *)srcp_u;
  const uint16_t *vptr = (const uint16_t *)srcp_v;
  uint32_t *outbuf = (uint32_t *)dstp;
  const int out_pitch = ((16 * ((width + 5) / 6) + 127) & ~127) / 4;
  for (int y = 0; y < height; y++) {
    const uint16_t *yline = yptr;
    const uint16_t *uline = uptr;
    const uint16_t *vline = vptr;
    uint32_t *out_line = outbuf;
    for (int x = 0; x < width + 5; x += 6) {
      out_line[0] = (uline[0] | (yline[0] << 10) | (vline[0] << 20));
      out_line[1] = (yline[1] | (uline[1] << 10) | (yline[2] << 20));
      out_line[2] = (vline[1] | (yline[3] << 10) | (uline[2] << 20));
      out_line[3] = (yline[4] | (vline[2] << 10) | (yline[5] << 20));
      out_line += 4;
      yline += 6;
      uline += 3;
      vline += 3;
    }
    outbuf += out_pitch;
    yptr += ppitch_y;
    uptr += ppitch_uv;
    vptr += ppitch_uv;
  }
}


void v210_to_yuv422p10(BYTE *dstp_y, int dstpitch, BYTE *dstp_u, BYTE *dstp_v, int dstpitch_uv, const BYTE *srcp, int width, int height)
{
  /*
  v210: packed YUV 4:2:2 (UYVY) 10 bits per component.
  Data is stored in blocks of 32 bit values in little-endian.
  Each such block contains 3 components, one each in bits 0 - 9, 10 - 19 and 20 - 29, the remaining two bits are unused.
  Six pixels in a pattern that repeats every 4 32-bits blocks:

  block 1, bits  0 -  9: U0-1
  block 1, bits 10 - 19: Y0
  block 1, bits 20 - 29: V0-1
  block 2, bits  0 -  9: Y1
  block 2, bits 10 - 19: U2-3
  block 2, bits 20 - 29: Y2
  block 3, bits  0 -  9: V2-3
  block 3, bits 10 - 19: Y3
  block 3, bits 20 - 29: U4-5
  block 4, bits  0 -  9: Y4
  block 4, bits 10 - 19: V4-5
  block 4, bits 20 - 29: Y5

  */
  int ppitch_y = dstpitch / sizeof(uint16_t);
  int ppitch_uv = dstpitch_uv / sizeof(uint16_t);
  uint16_t *yptr = (uint16_t *)dstp_y;
  uint16_t *uptr = (uint16_t *)dstp_u;
  uint16_t *vptr = (uint16_t *)dstp_v;

  const int srcpitch = ((16 * ((width + 5) / 6) + 127) & ~127);

  const int width6 = (width / 6) * 6;
  const int width_rest = width - width6;

  for (int y = 0; y < height; y++) {
    uint16_t *yline = yptr;
    uint16_t *uline = uptr;
    uint16_t *vline = vptr;
    const uint32_t *srcline = reinterpret_cast<const uint32_t *>(srcp);

    for (int x = 0; x < width6; x += 6) {
      uint32_t block1 = srcline[0];
      uint32_t block2 = srcline[1];
      uint32_t block3 = srcline[2];
      uint32_t block4 = srcline[3];

      uline[0] = (block1) & 0x3FF;
      vline[0] = (block1 >> 20) & 0x3FF;
      yline[0] = (block1 >> 10) & 0x3FF;
      yline[1] = (block2) & 0x3FF;

      uline[1] = (block2 >> 10) & 0x3FF;
      vline[1] = (block3) & 0x3FF;
      yline[2] = (block2 >> 20) & 0x3FF;
      yline[3] = (block3 >> 10) & 0x3FF;

      uline[2] = (block3 >> 20) & 0x3FF;
      vline[2] = (block4 >> 10) & 0x3FF;
      yline[4] = (block4) & 0x3FF;
      yline[5] = (block4 >> 20) & 0x3FF;

      srcline += 4;
      yline += 6;
      uline += 3;
      vline += 3;
    }
    // rest 2 or 4 pixels
    if (width_rest >= 2) {
      uint32_t block1 = srcline[0];
      uint32_t block2 = srcline[1];

      uline[0] = (block1) & 0x3FF;
      vline[0] = (block1 >> 20) & 0x3FF;
      yline[0] = (block1 >> 10) & 0x3FF;
      yline[1] = (block2) & 0x3FF;

      if (width_rest >= 4) {
        uint32_t block3 = srcline[2];

        uline[1] = (block2 >> 10) & 0x3FF;
        vline[1] = (block3) & 0x3FF;
        yline[2] = (block2 >> 20) & 0x3FF;
        yline[3] = (block3 >> 10) & 0x3FF;
      }
    }
    srcp += srcpitch;
    yptr += ppitch_y;
    uptr += ppitch_uv;
    vptr += ppitch_uv;
  }
}

void v410_to_yuv444p10(BYTE *dstp_y, int dstpitch, BYTE *dstp_u, BYTE *dstp_v, int dstpitch_uv, const BYTE *srcp, int width, int height)
{
  int ppitch_y = dstpitch / sizeof(uint16_t);
  int ppitch_uv = dstpitch_uv / sizeof(uint16_t);
  uint16_t *yptr = (uint16_t *)dstp_y;
  uint16_t *uptr = (uint16_t *)dstp_u;
  uint16_t *vptr = (uint16_t *)dstp_v;

  const int srcpitch = width * 2;

  for (int y = 0; y < height; y++) {
    uint16_t *yline = yptr;
    uint16_t *uline = uptr;
    uint16_t *vline = vptr;
    const uint32_t *srcline = reinterpret_cast<const uint32_t *>(srcp);

    for (int x = 0; x < width; x++) {
      uint32_t block = srcline[x];

      uline[x] = (block) & 0x3FF;
      vline[x] = (block >> 20) & 0x3FF;
      yline[x] = (block >> 10) & 0x3FF;
    }
    srcp += srcpitch;
    yptr += ppitch_y;
    uptr += ppitch_uv;
    vptr += ppitch_uv;
  }
}

void yuv42xp10_16_to_Px10_16(BYTE *dstp, int dstpitch, const BYTE *srcp_y, int srcpitch,
  const BYTE *srcp_u, const BYTE *srcp_v, int srcpitch_uv, 
  int width, int height, int cheight, bool semi_packed_p16, IScriptEnvironment *env)
{
  // P010/P016/P210/P216 format:
  // Single buffer
  // n lines   YYYYYYYYYYYYYY
  // n/2 lines UVUVUVUVUVUVUV (4:2:0)
  // or n lines UVUVUVUVUVUVUV (4:2:2)
  // Pitch is common. P010/P210 is upshifted to 16 bits
  
  const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);

  // luma
  if (semi_packed_p16) {
    // no shift, native copy
    env->BitBlt(dstp, dstpitch, srcp_y, srcpitch, width * sizeof(uint16_t), height);
  }
  else {
    // shift by 6 make 10->16 bits
    if (sse2)
      prepare_luma_shift6_sse2<true>(dstp, dstpitch, srcp_y, srcpitch, width, height); // true: conv to P016
    else
      prepare_luma_shift6_c<true>(dstp, dstpitch, srcp_y, srcpitch, width, height); // true: conv to P016
  }

  dstp += dstpitch * height;

  // Chroma
  int cwidth = width / 2;

  if (sse2) {
    if (semi_packed_p16)
      prepare_to_interleaved_uv_sse2<false>(dstp, dstpitch, srcp_u, srcp_v, srcpitch_uv, cwidth, cheight);
    else
      prepare_to_interleaved_uv_sse2<true>(dstp, dstpitch, srcp_u, srcp_v, srcpitch_uv, cwidth, cheight); // shift6 inside
  }
  else {
    if (semi_packed_p16)
      prepare_to_interleaved_uv_c<false>(dstp, dstpitch, srcp_u, srcp_v, srcpitch_uv, cwidth, cheight);
    else
      prepare_to_interleaved_uv_c<true>(dstp, dstpitch, srcp_u, srcp_v, srcpitch_uv, cwidth, cheight); // shift6 inside
  }

}

void Px10_16_to_yuv42xp10_16(BYTE *dstp_y, int dstpitch, BYTE *dstp_u, BYTE *dstp_v, int dstpitch_uv,
  const BYTE *srcp, int srcpitch,
  int width, int height, int cheight, bool semi_packed_p16, IScriptEnvironment *env)
{
  const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
  const bool sse41 = !!(env->GetCPUFlags() & CPUF_SSE4_1);

  // convert P010, P016, P210 and P216 formats back to Avisynth YUV420P10 and P16 or YUV422P10 and P16 formats

  // Luma
  if (semi_packed_p16) {
    env->BitBlt(dstp_y, dstpitch, srcp, srcpitch, width * sizeof(uint16_t), height);
  }
  else {
    // shift by 6 make 10->16 bits
    if (sse2)
      prepare_luma_shift6_sse2<false>(dstp_y, dstpitch, srcp, srcpitch, width, height); // false: after
    else
      prepare_luma_shift6_c<false>(dstp_y, dstpitch, srcp, srcpitch, width, height); // false: after
  }
  srcp += srcpitch * height;

  // Chroma
  int cwidth = width / 2; // 422 or 420
  if (sse41) {
    if (semi_packed_p16)
      prepare_from_interleaved_uv_sse2<false, true>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight);
    else
      prepare_from_interleaved_uv_sse2<true, true>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight); // true: shift 6
  }
  else if (sse2) {
    if (semi_packed_p16)
      prepare_from_interleaved_uv_sse2<false, false>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight);
    else
      prepare_from_interleaved_uv_sse2<true, false>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight); // true: shift 6
  }
  else {
    if (semi_packed_p16)
      prepare_from_interleaved_uv_c<false>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight);
    else
      prepare_from_interleaved_uv_c<true>(dstp_u, dstp_v, dstpitch_uv, srcp, srcpitch, cwidth, cheight); // true: shift 6
  }
}

