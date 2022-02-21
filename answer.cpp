#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <ostream>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#ifndef NDEBUG
#warning "NDEBUG マクロが定義されてないよ！動作が遅くなるかもしれないよ！"
#endif

#ifdef __GNUC__
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("arch=skylake"))), apply_to = function)
/* 最後に↓を貼る
#ifdef __clang__
#pragma clang attribute pop
#endif
*/
#else // defined(__clang__)
//#pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,fma,avx,avx2,tune=native")
//#pragma GCC optimize("unroll-loops")
#endif // defined(__clang__)
#endif // defined(__GNUC__)

using namespace std;

namespace nn {

template <class T, int dim1, int dim2 = -1, int dim3 = -1, int dim4 = -1> struct Tensor {
    // データは常に contiguous で持つ
    // データ数が int に収まることを仮定
    using value_type = T;

    constexpr static auto n_data = max(1, dim1) * max(1, dim2) * max(1, dim3) * max(1, dim4); // これ 0 のときにおかしくなるな
    constexpr static auto rank = dim1 == -1 ? 0 : dim2 == -1 ? 1 : dim3 == -1 ? 2 : dim4 == -1 ? 3 : 4;
    array<T, n_data> data;

    constexpr inline auto begin() {
        if constexpr (rank == 1)
            return &data[0];
        else {
            union U {
                Tensor<T, dim1, dim2, dim3, dim4> current;
                Tensor<T, dim2, dim3, dim4> data_i;
            };
            return &((U*)this)->data_i;
        }
    }
    constexpr inline auto begin() const {
        if constexpr (rank == 1)
            return &data[0];
        else {
            union U {
                Tensor<T, dim1, dim2, dim3, dim4> current;
                Tensor<T, dim2, dim3, dim4> data_i;
            };
            return &((U*)this)->data_i;
        }
    }
    constexpr inline auto end() { return begin() + dim1; }
    constexpr inline auto end() const { return begin() + dim1; }
    auto& operator[](const int& idx) {
        assert(0 <= idx && idx < dim1);
        return begin()[idx];
    }
    const auto& operator[](const int& idx) const {
        assert(0 <= idx && idx < dim1);
        return begin()[idx];
    }
    auto& Ravel() {
        union U {
            Tensor<T, dim1, dim2, dim3, dim4> current;
            Tensor<T, n_data> raveled;
        };
        return ((U*)this)->raveled;
    }
    const auto& Ravel() const {
        union U {
            Tensor<T, dim1, dim2, dim3, dim4> current;
            Tensor<T, n_data> raveled;
        };
        return ((U*)this)->raveled;
    }
    template <int new_dim_1, int new_dim_2 = -1, int new_dim_3 = -1, int new_dim_4 = -1> auto& View() {
        using ResultType = Tensor<T, new_dim_1, new_dim_2, new_dim_3, new_dim_4>;
        static_assert(n_data == ResultType::n_data, "View の次元がおかしいよ");
        union U {
            Tensor data;
            ResultType view;
        };
        return ((U*)this)->view;
    }
    void Fill(const T& fill_value) { fill(&data[0], (&data[0]) + n_data, fill_value); }
    void Iota(const T& start_value) { iota(&data[0], (&data[0]) + n_data, start_value); }

    constexpr static int Shape(const int& d) { return d == 0 ? dim1 : d == 1 ? dim2 : d == 2 ? dim3 : d == 3 ? dim4 : -1; }
    template <int p0, int p1 = -1, int p2 = -1, int p3 = -1> auto Permute() const {
        // 転置してコピーを返す
        // d0 が 1 の場合、もともと 1 次元目だったところが 0 次元目に来る
        auto indices = array<int, rank>();
        auto permuted = Tensor<T, Shape(p0), Shape(p1), Shape(p2), Shape(p3)>();
        if constexpr (rank == 2) {
            for (indices[0] = 0; indices[0] < dim1; indices[0]++)
                for (indices[1] = 0; indices[1] < dim2; indices[1]++)
                    permuted[indices[p0]][indices[p1]] = (*this)[indices[0]][indices[1]];
        } else if constexpr (rank == 3) {
            for (indices[0] = 0; indices[0] < dim1; indices[0]++)
                for (indices[1] = 0; indices[1] < dim2; indices[1]++)
                    for (indices[2] = 0; indices[2] < dim3; indices[2]++)
                        permuted[indices[p0]][indices[p1]][indices[p2]] = (*this)[indices[0]][indices[1]][indices[2]];
        } else if constexpr (rank == 4) {
            for (indices[0] = 0; indices[0] < dim1; indices[0]++)
                for (indices[1] = 0; indices[1] < dim2; indices[1]++)
                    for (indices[2] = 0; indices[2] < dim3; indices[2]++)
                        for (indices[3] = 0; indices[3] < dim4; indices[3]++)
                            permuted[indices[p0]][indices[p1]][indices[p2]][indices[p3]] = (*this)[indices[0]][indices[1]][indices[2]][indices[3]];
        } else {
            assert(false);
        }
        return permuted;
    }
    inline void Clamp_(const T& min_value, const T& max_value) {
        if constexpr (is_same<T, int>() && n_data % 8 == 0) {
            assert(((intptr_t)this & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_min_epi32(*(__m256i*)&data[i], _mm256_set1_epi32(max_value));
                *(__m256i*)&data[i] = _mm256_max_epi32(*(__m256i*)&data[i], _mm256_set1_epi32(min_value));
            }
        } else if constexpr (is_same<T, short>() && n_data % 16 == 0) {
            assert(((intptr_t)this & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 16) {
                *(__m256i*)&data[i] = _mm256_min_epi16(*(__m256i*)&data[i], _mm256_set1_epi16(max_value));
                *(__m256i*)&data[i] = _mm256_max_epi16(*(__m256i*)&data[i], _mm256_set1_epi16(min_value));
            }
        } else {
            // 愚直
            for (auto&& x : data)
                x = clamp(x, min_value, max_value);
        }
    }
    template <typename S> inline auto& operator+=(const Tensor<S, dim1, dim2, dim3, dim4>& rhs) {
        if constexpr (is_same<T, int>() && is_same<S, int>() && n_data % 8 == 0) {
            // 飽和しない
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_add_epi32(*(__m256i*)&data[i], *(__m256i*)&rhs.data[i]);
            }
        } else {
            // 愚直
            for (int i = 0; i < n_data; i++) {
                assert(numeric_limits<T>::max() - data[i] >= rhs.data[i]); // オーバーフロー確認
                data[i] += rhs.data[i];
            }
        }
        return *this;
    }
    template <typename S> inline auto& Adds_(const Tensor<S, dim1, dim2, dim3, dim4>& rhs) {
        static_assert(is_same<T, short>() && is_same<S, short>());
        if constexpr (is_same<T, short>() && is_same<S, short>() && n_data % 16 == 0) {
            // 飽和する
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 16) {
                *(__m256i*)&data[i] = _mm256_adds_epi16(*(__m256i*)&data[i], *(__m256i*)&rhs.data[i]);
            }
        } else {
            // 愚直
            // 飽和しない でいいかと思ったけどこれ気づかないので良くないな
            assert(false);
            for (int i = 0; i < n_data; i++) {
                // assert(numeric_limits<T>::max() - data[i] >= rhs.data[i]); // オーバーフロー確認
                data[i] += rhs.data[i];
            }
        }
        return *this;
    }
    inline auto& operator+=(const T& rhs) {
        if constexpr (is_same<T, int>() && n_data % 8 == 0) {
            // 飽和しない
            assert(((intptr_t)this & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_add_epi32(_mm256_set1_epi32(rhs), *(__m256i*)&data[i]);
            }
        } else {
            // 愚直
            for (int i = 0; i < n_data; i++)
                data[i] += rhs;
        }
        return *this;
    }
    auto& SubtractFrom_(const T& rhs) {
        if constexpr (is_same<T, int>() && n_data % 8 == 0) {
            // 飽和しない
            assert(((intptr_t)this & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_sub_epi32(_mm256_set1_epi32(rhs), *(__m256i*)&data[i]);
            }
        } else {
            // 愚直
            for (int i = 0; i < n_data; i++)
                data[i] = rhs - data[i];
        }
        return *this;
    }
    template <typename S> inline auto& operator*=(const Tensor<S, dim1, dim2, dim3, dim4>& rhs) {
        if constexpr (is_same<T, int>() && is_same<S, int>() && n_data % 8 == 0) {
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_mullo_epi32(*(__m256i*)&data[i], *(__m256i*)&rhs.data[i]);
            }
        } else {
            // 愚直
            for (int i = 0; i < n_data; i++)
                data[i] *= rhs.data[i];
        }
        return *this;
    }
    inline auto& operator*=(const T& rhs) {
        // 愚直
        for (int i = 0; i < n_data; i++)
            data[i] *= rhs;
        return *this;
    }
    inline auto& operator>>=(const int& rhs) {
        if constexpr (is_same<T, int>() && n_data % 8 == 0) {
            assert(((intptr_t)this & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 8) {
                *(__m256i*)&data[i] = _mm256_srai_epi32(*(__m256i*)&data[i], rhs);
            }
        } else {
            // 愚直
            for (int i = 0; i < n_data; i++)
                data[i] >>= rhs;
        }
        return *this;
    }
    template <typename S> inline auto& operator=(const Tensor<S, dim1, dim2, dim3, dim4>& rhs) {
        if constexpr (is_same<T, signed char>() && is_same<S, int>() && n_data % 32 == 0) {
            // 飽和する
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 32) {
                const auto p1 = _mm256_packs_epi32(*(__m256i*)&rhs.data[i], *(__m256i*)&rhs.data[i + 8]);       // 0213
                const auto p2 = _mm256_packs_epi32(*(__m256i*)&rhs.data[i + 16], *(__m256i*)&rhs.data[i + 24]); // 4657
                const auto p = _mm256_packs_epi16(p1, p2);                                                      // 02461357
                *(__m256i*)&data[i] = _mm256_permutevar8x32_epi32(p, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)); // 01234567
            }
        } else if constexpr (is_same<T, unsigned char>() && is_same<S, int>() && n_data % 32 == 0) {
            // 飽和する
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 32) {
                const auto p1 = _mm256_packs_epi32(*(__m256i*)&rhs.data[i], *(__m256i*)&rhs.data[i + 8]);       // 0213
                const auto p2 = _mm256_packs_epi32(*(__m256i*)&rhs.data[i + 16], *(__m256i*)&rhs.data[i + 24]); // 4657
                const auto p = _mm256_packus_epi16(p1, p2);                                                     // 02461357
                *(__m256i*)&data[i] = _mm256_permutevar8x32_epi32(p, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0)); // 01234567
            }
        } else if constexpr (is_same<T, short>() && is_same<S, int>() && n_data % 16 == 0) {
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 16) {
                const auto p = _mm256_packs_epi32(*(__m256i*)&rhs.data[i], *(__m256i*)&rhs.data[i + 8]); // 0213
                *(__m256i*)&data[i] = _mm256_permute4x64_epi64(p, 0b11011000);                           // 0123
            }
        } else if constexpr (is_same<T, signed char>() && is_same<S, short>() && n_data % 32 == 0) {
            // 飽和する
            assert(((intptr_t)this & 0b11111) == 0);
            assert(((intptr_t)&rhs & 0b11111) == 0);
            for (int i = 0; i < n_data; i += 32) {
                const auto p = _mm256_packs_epi16(*(__m256i*)&rhs.data[i], *(__m256i*)&rhs.data[i + 16]); // 0213
                *(__m256i*)&data[i] = _mm256_permute4x64_epi64(p, 0b11011000);                            // 0123
            }
        } else if constexpr (is_same<T, S>()) {
            data = rhs.data;
        } else {
            for (int i = 0; i < n_data; i++)
                data[i] = (T)rhs.data[i];
        }
        return *this;
    }
    constexpr inline int size() const { return dim1; }

    void Print(ostream& os = cout) const {
        PrintImpl(os);
        os << endl;
    }
    void PrintImpl(ostream& os = cout) const {
        // os << "PrintImpl" << (intptr_t) &
        if constexpr (rank == 1) {
            for (int i = 0; i < dim1; i++) {
                os << (sizeof(data[i]) == 1 ? (int)data[i] : data[i]) << (i != dim1 - 1 ? " " : "");
            }
        } else {
            for (int i = 0; i < dim1; i++) {
                this->operator[](i).PrintImpl(os);
                if (i != dim1 - 1)
                    for (int j = 0; j < rank - 1; j++)
                        os << endl;
            }
        }
    }
    friend auto& operator<<(ostream& os, const Tensor& t) {
        t.Print(os);
        return os;
    }
};

namespace F {
template <class Tensor_> inline void Relu_(Tensor_& input) {
    if constexpr (is_same<typename Tensor_::value_type, signed char>() && Tensor_::n_data % 32 == 0) {
        assert(((intptr_t)&input & 0b11111) == 0);
        for (int i = 0; i < Tensor_::n_data; i += 32) {
            *(__m256i*)&input[i] = _mm256_max_epi8(*(__m256i*)&input[i], _mm256_setzero_si256());
        }
    } else {
        for (auto&& value : input.data) {
            value = max(value, (typename Tensor_::value_type)0);
        }
    }
}
template <class Tensor_> inline void Softmax_(Tensor_& input) {
    auto ma = numeric_limits<float>::min();
    for (const auto& v : input)
        if (ma < v)
            ma = v;
    auto s = 0.0f;
    for (const auto& v : input)
        s += expf(v - ma);
    auto c = ma + logf(s);
    for (auto&& v : input)
        v = expf(v - c);
}
template <class Tensor_> inline void Tanh_(Tensor_& input) {
    constexpr auto USE_AVX2 = true;
    if constexpr (USE_AVX2 && is_same<typename Tensor_::value_type, int>()) {
        // scale 13 -> 7
        // |x| <= 3 の範囲で近似

        static_assert(Tensor_::n_data % 8 == 0);
        assert(((intptr_t)&input & 0b11111) == 0);
        for (int i = 0; i < Tensor_::n_data; i += 8) {
            auto x = *(__m256i*)&input[i];
            x = _mm256_abs_epi32(x);
            x = _mm256_min_epi32(x, _mm256_set1_epi32(3 << 13));                    // [0, 24576] scale 13
            auto res = _mm256_set1_epi32((int)(0.0040006136474788475 * (1 << 17))); // 524
            for (auto&& coef :
                 {0.037782514883135755, -0.4164943773790996, 1.137696148833152, -0.007219235770736496 + 1.0 / 256.0}) { // 1 / 256 は切り捨て対策
                res = _mm256_mullo_epi32(res, x);                                                                       // scale 17 x 13 -> 30
                res = _mm256_add_epi32(res, _mm256_set1_epi32((int)(coef * (1 << 30))));                                // scale 30
                res = _mm256_srai_epi32(res, 13);                                                                       // scale 30 -> 17
            }
            res = _mm256_srai_epi32(res, 10); // scale 17 -> 7

            *(__m256i*)&input[i] = _mm256_sign_epi32(res, *(__m256i*)&input[i]);
        }
    } else if constexpr (is_same<typename Tensor_::value_type, int>()) {
        for (auto&& value : input.data)
            value = (int)(128.0f * tanhf((float)value / (float)(1 << 13)));
    } else if constexpr (is_same<typename Tensor_::value_type, float>()) {
        for (auto&& value : input.data)
            value = tanhf(value);
    } else {
        assert(false);
    }
}
template <int in_scale, int out_scale> struct TanhTable {
    static_assert(out_scale < 15);
    constexpr static auto table_size = 8 << in_scale;
    constexpr static auto x_min = -table_size / 2;
    constexpr static auto x_max = table_size / 2 - 1;
    array<short, table_size> data_buffer;
    short* data;
    inline TanhTable() {
        // 中に exp があるせいで constexpr 指定できない
        // これ複数回初期化される可能性あるけど、どうしたら綺麗に回避できる？
        data = &data_buffer[table_size / 2];
        for (int x = -table_size / 2; x < table_size / 2; x++) {
            data[x] = (short)round(tanhf((float)x / (float)(1 << in_scale)) * (1 << out_scale));
        }
    }
    inline const auto& operator[](const int& idx) const {
        assert(-table_size / 2 <= idx && idx < table_size / 2);
        return data[idx];
    }
};
template <class Tensor1, class Tensor2, int in_scale, int out_scale> inline void Tanh(const Tensor1& input, Tensor2& output) {
    // 表引きする
    // |x| <= 8 の範囲
    static_assert(Tensor1::n_data == Tensor2::n_data);
    static auto table = TanhTable<in_scale, out_scale>();
    for (int i = 0; i < Tensor1::n_data; i++) {
        output.Ravel()[i] = table[clamp(input.Ravel()[i], decltype(table)::x_min, decltype(table)::x_max)];
    }
}
template <class Tensor_> inline void Sigmoid_(Tensor_& input) {
    constexpr auto USE_AVX2 = true;
    if constexpr (USE_AVX2 && is_same<typename Tensor_::value_type, int>()) {
        // scale 13 -> 7
        // |x| <= 6 の範囲で近似

        static_assert(Tensor_::n_data % 8 == 0);
        assert(((intptr_t)&input & 0b11111) == 0);
        for (int i = 0; i < Tensor_::n_data; i += 8) {
            auto x = *(__m256i*)&input[i];
            x = _mm256_abs_epi32(x);
            x = _mm256_srai_epi32(x, 1);
            x = _mm256_min_epi32(x, _mm256_set1_epi32(3 << 13));                    // [0, 24384] scale 13
            auto res = _mm256_set1_epi32((int)(0.0040006136474788475 * (1 << 17))); // 536
            for (auto&& coef : {0.037782514883135755, -0.4164943773790996, 1.137696148833152, -0.007219235770736496}) {
                res = _mm256_mullo_epi32(res, x);                                        // scale 17 x 13 -> 30
                res = _mm256_add_epi32(res, _mm256_set1_epi32((int)(coef * (1 << 30)))); // scale 30
                res = _mm256_srai_epi32(res, 13);                                        // scale 30 -> 17
            }
            res = _mm256_sign_epi32(res, *(__m256i*)&input[i]);
            res = _mm256_add_epi32(res, _mm256_set1_epi32((1 << 17) + (1 << 10))); // +1 と切り捨て対策
            res = _mm256_srai_epi32(res, 11);                                      // 1/2 と scale 17 -> 7
            *(__m256i*)&input[i] = res;
        }
    } else if constexpr (is_same<typename Tensor_::value_type, int>()) {
        for (auto&& value : input.data)
            value = (int)(128.0f / (1.0f + expf((float)-value / (float)(1 << 13))));
    } else if constexpr (is_same<typename Tensor_::value_type, float>()) {
        for (auto&& value : input.data)
            value = 1.0f / (1.0f + expf(-value));
    } else {
        assert(false);
    }
}
template <int in_scale, int out_scale> struct SigmoidTable {
    static_assert(out_scale < 15);
    constexpr static auto table_size = 16 << in_scale;
    constexpr static auto x_min = -table_size / 2;
    constexpr static auto x_max = table_size / 2 - 1;
    array<short, table_size> data_buffer;
    short* data;
    inline SigmoidTable() {
        // 中に exp があるせいで constexpr 指定できない
        // これ複数回初期化される可能性あるけど、どうしたら綺麗に回避できる？
        data = &data_buffer[table_size / 2];
        for (int x = -table_size / 2; x < table_size / 2; x++) {
            data[x] = (short)round((1.0f / (1.0f + expf((float)-x / (float)(1 << in_scale)))) * (1 << out_scale));
        }
    }
    inline const auto& operator[](const int& idx) const {
        assert(-table_size / 2 <= idx && idx < table_size / 2);
        return data[idx];
    }
};
template <class Tensor1, class Tensor2, int in_scale, int out_scale> inline void Sigmoid(const Tensor1& input, Tensor2& output) {
    // 表引きする
    // |x| <= 8 の範囲
    static_assert(Tensor1::n_data == Tensor2::n_data);
    static auto table = SigmoidTable<in_scale, out_scale>();
    for (int i = 0; i < Tensor1::n_data; i++) {
        output.Ravel()[i] = table[clamp(input.Ravel()[i], decltype(table)::x_min, decltype(table)::x_max)];
    }
}

} // namespace F

template <int in_features, int out_features, typename dtype = float, typename in_dtype = float, typename out_dtype = float, int weight_scale_ = -1,
          int in_scale_ = -1, int out_scale_ = -1>
struct alignas(32) Linear {
    // scale 7 -> 13
    // 掛け算で値が飽和することがあるので、重みの値が大きいときは注意

    constexpr static auto dtypes_condition_1 = is_same<dtype, float>() && is_same<in_dtype, float>() && is_same<out_dtype, float>();
    constexpr static auto dtypes_condition_2 = is_same<dtype, signed char>() && is_same<in_dtype, signed char>() && is_same<out_dtype, int>();
    constexpr static auto dtypes_condition_3 = is_same<dtype, signed char>() && is_same<in_dtype, unsigned char>() && is_same<out_dtype, int>();
    constexpr static auto dtypes_condition_4 = is_same<dtype, short>() && is_same<in_dtype, short>() && is_same<out_dtype, int>();
    static_assert(dtypes_condition_1 || dtypes_condition_2 || dtypes_condition_3 || dtypes_condition_4);

    constexpr static auto weight_scale =
        weight_scale_ != -1 ? weight_scale_
                            : is_same<dtype, float>() ? 0 : is_same<dtype, short>() ? 12 : in_features >= 256 ? 8 : in_features >= 64 ? 7 : 6;
    constexpr static auto in_scale = in_scale_ != -1 ? in_scale_ : is_same<in_dtype, float>() ? 0 : is_same<in_dtype, short>() ? 12 : 7;
    constexpr static auto out_scale =
        out_scale_ != -1 ? out_scale_
                         : is_same<out_dtype, float>() ? 0 : 13; // 注: bias のスケールとは異なる。bias のスケールは weight_scale と in_scale の和。

    static_assert(weight_scale + in_scale >= out_scale);

    // パラメータ
    Tensor<dtype, out_features, in_features> weight;
    Tensor<out_dtype, out_features> bias;

    // コンストラクタ
    Linear() = default;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        if (fread(&weight, sizeof(dtype), weight.n_data, f) != weight.n_data)
            abort();
        if (fread(&bias, sizeof(out_dtype), bias.n_data, f) != bias.n_data)
            abort();
        if constexpr (is_same<in_dtype, signed char>()) {
            SlideBias();
        }
    }

    void SlideBias() {
        // intrinsic 関数の都合上、入力は unsigned であってほしいが、今は signed なので、
        // バイアスを予め調整しておいて入力には 128 を足す (符号ビットを反転させる) ことで対応する
        static_assert(is_same<out_dtype, int>());
        for (int o = 0; o < out_features; o++) {
            for (int i = 0; i < in_features; i++) {
                bias[o] -= 128 * (int)weight[o][i];
            }
        }
    }

    // 順伝播
    inline void Forward(const Tensor<in_dtype, in_features>& input, Tensor<out_dtype, out_features>& output) const {
        constexpr static auto USE_AVX = true;
        constexpr static auto USE_AVX2 = true;

        if constexpr (USE_AVX2 && sizeof(in_dtype) == 1 && is_same<out_dtype, int>() && out_features % 4 == 0 && in_features % 32 == 0) {

            // 参考:
            // https://github.com/yaneurao/YaneuraOu/blob/f94720b9b72aaa992b02e45914590c63b3d114b2/source/eval/nnue/layers/affine_transform.h

            assert(((intptr_t)&input & 0b11111) == 0);
            assert(((intptr_t)&output & 0b11111) == 0);
            assert(((intptr_t)&weight & 0b11111) == 0);
            assert(((intptr_t)&bias & 0b11111) == 0);

            const __m256i kOnes256 = _mm256_set1_epi16(1);
            static constexpr auto kSimdWidth = (int)(sizeof(__m256i) / sizeof(dtype)); // 32 / 1 = 32
            static constexpr auto kNumChunks = in_features / kSimdWidth;               // in_features が 256 なら 8, 32 なら 1

            // 内積を計算したい
            // 256 bit = 8 bit x 32 のベクトル a, b から、a[i] * b[i] を計算してちょっと集約して 32 bit x 8 にする
            // a は unsigned
            auto m256_add_dpbusd_epi32 = [=](__m256i& acc, __m256i a, __m256i b) {
                __m256i product0 = _mm256_maddubs_epi16(a, b);    //  8 bit x 32 -> 16 bit x 16  入力が 129 以上の場合、飽和し得る
                product0 = _mm256_madd_epi16(product0, kOnes256); // 16 bit x 16 -> 32 bit x  8
                acc = _mm256_add_epi32(acc, product0);
            };

            // 32 bit x 8 のベクトル 4 つをそれぞれ集約して 1 つの 32 bit x 4 のベクトルにする
            auto m256_haddx4 = [](__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) -> __m128i {
                sum0 = _mm256_hadd_epi32(sum0, sum1); // 00110011
                sum2 = _mm256_hadd_epi32(sum2, sum3); // 22332233

                sum0 = _mm256_hadd_epi32(sum0, sum2); // 01230123

                __m128i sum128lo = _mm256_castsi256_si128(sum0);
                __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

                return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias); // 0123
            };

            const __m256i* input_vector;
            alignas(32) static auto input_ = Tensor<unsigned char, in_features>();
            if constexpr (is_same<in_dtype, signed char>()) {
                // intrinsic 関数の都合上、入力は unsigned であってほしいが、今は signed なので、
                // バイアスを予め調整しておいて入力には 128 を足す (符号ビットを反転させる) ことで対応する
                input_vector = (const __m256i*)(&input_[0]);
                for (int i = 0; i < in_features; i += 32) {
                    *(__m256i*)&input_[i] = _mm256_xor_si256(*(__m256i*)&input[i], _mm256_set1_epi8(0b10000000));
                }
            } else {
                input_vector = (const __m256i*)(&input[0]);
            }

            for (int i = 0; i < out_features; i += 4) {
                const __m128i bias_ = *(const __m128i*)(&bias[i]);
                __m128i* outptr = (__m128i*)(&output[i]);

                __m256i sum0 = _mm256_setzero_si256();
                __m256i sum1 = _mm256_setzero_si256();
                __m256i sum2 = _mm256_setzero_si256();
                __m256i sum3 = _mm256_setzero_si256();

                const auto row0 = (const __m256i*)(&weight[i]);
                const auto row1 = (const __m256i*)(&weight[i + 1]);
                const auto row2 = (const __m256i*)(&weight[i + 2]);
                const auto row3 = (const __m256i*)(&weight[i + 3]);

                for (int j = 0; j < kNumChunks; j++) {
                    const __m256i in = input_vector[j];

                    m256_add_dpbusd_epi32(sum0, in, row0[j]);
                    m256_add_dpbusd_epi32(sum1, in, row1[j]);
                    m256_add_dpbusd_epi32(sum2, in, row2[j]);
                    m256_add_dpbusd_epi32(sum3, in, row3[j]);
                }
                *outptr = m256_haddx4(sum0, sum1, sum2, sum3, bias_);
            }
            if constexpr (weight_scale > 6) {
                for (int o = 0; o < out_features; o += 8) {
                    *(__m256i*)&output[o] = _mm256_srai_epi32(*(__m256i*)&output[o], weight_scale - 6);
                }
            }
        } else if constexpr (USE_AVX2 && sizeof(in_dtype) == 2 && is_same<out_dtype, int>() && out_features % 4 == 0 && in_features % 16 == 0) {
            // 16 bit で量子化する場合
            // 2^12 倍にスケーリングする
            assert(((intptr_t)&input & 0b11111) == 0);
            assert(((intptr_t)&output & 0b11111) == 0);
            assert(((intptr_t)&weight & 0b11111) == 0);
            assert(((intptr_t)&bias & 0b11111) == 0);

            for (int o = 0; o < out_features; o += 4) {
                auto sum0 = _mm256_setzero_si256(); // 32 bit x 8
                auto sum1 = _mm256_setzero_si256();
                auto sum2 = _mm256_setzero_si256();
                auto sum3 = _mm256_setzero_si256();
                for (int i = 0; i < in_features; i += 16) {
                    const auto in = *(const __m256i*)&input[i];
                    sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(in, *(const __m256i*)&weight[o][i]));
                    sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(in, *(const __m256i*)&weight[o + 1][i]));
                    sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(in, *(const __m256i*)&weight[o + 2][i]));
                    sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(in, *(const __m256i*)&weight[o + 3][i]));
                }
                sum0 = _mm256_hadd_epi32(sum0, sum1);             // 00110011
                sum2 = _mm256_hadd_epi32(sum2, sum3);             // 22332233
                sum0 = _mm256_hadd_epi32(sum0, sum2);             // 01230123
                auto sum0_lo = _mm256_castsi256_si128(sum0);      // 0123
                auto sum0_hi = _mm256_extracti128_si256(sum0, 1); // 0123

                auto& out = *(__m128i*)(&output[o]);
                out = *(const __m128i*)(&bias[o]);
                out = _mm_add_epi32(out, _mm_add_epi32(sum0_lo, sum0_hi));
                out = _mm_srai_epi32(out, weight_scale + in_scale - out_scale);
            }
        } else if constexpr (USE_AVX && is_same<dtype, float>() && out_features % 4 == 0 && in_features % 8 == 0) {
            assert(((intptr_t)&input & 0b11111) == 0);
            assert(((intptr_t)&output & 0b11111) == 0);
            assert(((intptr_t)&weight & 0b11111) == 0);
            assert(((intptr_t)&bias & 0b11111) == 0);

            for (int o = 0; o < out_features; o += 4) {
                auto sum0 = _mm256_setzero_ps(); // 32 bit x 8
                auto sum1 = _mm256_setzero_ps();
                auto sum2 = _mm256_setzero_ps();
                auto sum3 = _mm256_setzero_ps();
                for (int i = 0; i < in_features; i += 8) {
                    const auto in = *(const __m256*)&input[i];
                    sum0 = _mm256_fmadd_ps(in, *(const __m256*)&weight[o][i], sum0);
                    sum1 = _mm256_fmadd_ps(in, *(const __m256*)&weight[o + 1][i], sum1);
                    sum2 = _mm256_fmadd_ps(in, *(const __m256*)&weight[o + 2][i], sum2);
                    sum3 = _mm256_fmadd_ps(in, *(const __m256*)&weight[o + 3][i], sum3);
                }
                sum0 = _mm256_hadd_ps(sum0, sum1);             // 00110011
                sum2 = _mm256_hadd_ps(sum2, sum3);             // 22332233
                sum0 = _mm256_hadd_ps(sum0, sum2);             // 01230123
                auto sum0_lo = _mm256_castps256_ps128(sum0);   // 0123
                auto sum0_hi = _mm256_extractf128_ps(sum0, 1); // 0123

                auto& out = *(__m128*)(&output[o]);
                out = *(const __m128*)(&bias[o]);
                out = _mm_add_ps(out, _mm_add_ps(sum0_lo, sum0_hi));
            }
        } else {
            output = bias;
            for (int o = 0; o < out_features; o++) {
                for (int i = 0; i < in_features; i++) {
                    if constexpr (is_same<in_dtype, signed char>()) {
                        if (output[o] > 0 && numeric_limits<out_dtype>::max() - output[o] <
                                                 (out_dtype)(unsigned char)(input[i] ^ 0b10000000) * (out_dtype)weight[o][i]) {
                            // オーバーフロー
                            cout << "オーバーフロー!!!!!" << endl;
                            cout << "output[o]=" << output[o] << endl;
                            cout << "(out_dtype)(unsigned char)(input[i] ^ 0b10000000)=" << (out_dtype)(unsigned char)(input[i] ^ 0b10000000) << endl;
                            cout << "(out_dtype)weight[o][i]=" << (out_dtype)weight[o][i] << endl;
                            assert(numeric_limits<out_dtype>::max() - output[o] >=
                                   (out_dtype)(unsigned char)(input[i] ^ 0b10000000) * (out_dtype)weight[o][i]);
                        }

                        output[o] += (out_dtype)(unsigned char)(input[i] ^ 0b10000000) * (out_dtype)weight[o][i];
                    } else {
                        assert(output[o] <= 0 || numeric_limits<out_dtype>::max() - output[o] >= (out_dtype)input[i] * (out_dtype)weight[o][i]);
                        output[o] += (out_dtype)input[i] * (out_dtype)weight[o][i];
                    }
                }
                if constexpr (is_same<out_dtype, int>() && weight_scale + in_scale - out_scale > 0) {
                    output[o] >>= weight_scale + in_scale - out_scale;
                }
            }
        }
    }
};

template <int num_embeddings, int embedding_dim, typename dtype = float> struct alignas(64) EmbeddingBag {
    // mode="sum" のみ対応
    // スケーリングは任意

    // パラメータ
    Tensor<dtype, num_embeddings, embedding_dim> weight;

    // コンストラクタ
    EmbeddingBag() : weight() {}

    // パラメータ読み込み
    void ReadParameters(FILE* f) {
        if (fread(&weight, sizeof(dtype), weight.n_data, f) != weight.n_data)
            abort();
    }

    // 順伝播
    template <class Container> inline void Forward(const Container& input, Tensor<dtype, embedding_dim>& output) const {
        constexpr static auto USE_AVX2 = true;

        output.Fill((dtype)0);
        if constexpr (USE_AVX2 && is_same<dtype, short>() && embedding_dim % 16 == 0) {
            assert(((intptr_t)&output & 0b11111) == 0);
            assert(((intptr_t)&weight & 0b11111) == 0);

            static constexpr auto kSimdWidth = (int)(sizeof(__m256i) / sizeof(dtype)); // 32 / 2 = 16
            static constexpr auto kNumChunks = embedding_dim / kSimdWidth;             // 16
            const auto out_ptr = (__m256i*)(&output[0]);

            for (const auto& idx : input) {
                const auto weight_column = (const __m256i*)(&weight[idx][0]);
                for (int chunk = 0; chunk < kNumChunks; chunk++) {
                    out_ptr[chunk] = _mm256_adds_epi16(out_ptr[chunk], weight_column[chunk]);
                }
            }
        } else {
            for (const auto& idx : input) {
                for (int dim = 0; dim < embedding_dim; dim++) {
                    output[dim] += weight[idx][dim];
                }
            }
        }
    }
};

template <int input_size, int hidden_size, typename dtype = float, typename in_dtype = float, typename out_dtype = float> struct alignas(32) GRU {
    // 実際は GRUCell 相当
    // scale 7 -> 7 または 12 -> 12

    constexpr static auto dtypes_condition_1 = is_same<dtype, float>() && is_same<in_dtype, float>() && is_same<out_dtype, float>();
    constexpr static auto dtypes_condition_2 = is_same<dtype, signed char>() && is_same<in_dtype, signed char>() && is_same<out_dtype, int>();
    constexpr static auto dtypes_condition_3 = is_same<dtype, short>() && is_same<in_dtype, short>() && is_same<out_dtype, int>();
    static_assert(dtypes_condition_1 || dtypes_condition_2 || dtypes_condition_3);

    // パラメータ
    Linear<input_size, hidden_size, dtype, in_dtype, out_dtype> w_ir;
    Linear<hidden_size, hidden_size, dtype, dtype, out_dtype> w_hr;
    Linear<input_size, hidden_size, dtype, in_dtype, out_dtype> w_iz;
    Linear<hidden_size, hidden_size, dtype, dtype, out_dtype> w_hz;
    Linear<input_size, hidden_size, dtype, in_dtype, out_dtype> w_in;
    Linear<hidden_size, hidden_size, dtype, dtype, out_dtype> w_hn;

    // コンストラクタ
    GRU() = default;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        if (fread(&w_ir.weight, sizeof(dtype), w_ir.weight.n_data, f) != w_ir.weight.n_data)
            abort();
        if (fread(&w_iz.weight, sizeof(dtype), w_iz.weight.n_data, f) != w_iz.weight.n_data)
            abort();
        if (fread(&w_in.weight, sizeof(dtype), w_in.weight.n_data, f) != w_in.weight.n_data)
            abort();
        if (fread(&w_hr.weight, sizeof(dtype), w_hr.weight.n_data, f) != w_hr.weight.n_data)
            abort();
        if (fread(&w_hz.weight, sizeof(dtype), w_hz.weight.n_data, f) != w_hz.weight.n_data)
            abort();
        if (fread(&w_hn.weight, sizeof(dtype), w_hn.weight.n_data, f) != w_hn.weight.n_data)
            abort();
        if (fread(&w_ir.bias, sizeof(out_dtype), w_ir.bias.n_data, f) != w_ir.bias.n_data)
            abort();
        if (fread(&w_iz.bias, sizeof(out_dtype), w_iz.bias.n_data, f) != w_iz.bias.n_data)
            abort();
        if (fread(&w_in.bias, sizeof(out_dtype), w_in.bias.n_data, f) != w_in.bias.n_data)
            abort();
        if (fread(&w_hr.bias, sizeof(out_dtype), w_hr.bias.n_data, f) != w_hr.bias.n_data)
            abort();
        if (fread(&w_hz.bias, sizeof(out_dtype), w_hz.bias.n_data, f) != w_hz.bias.n_data)
            abort();
        if (fread(&w_hn.bias, sizeof(out_dtype), w_hn.bias.n_data, f) != w_hn.bias.n_data)
            abort();
        if constexpr (is_same<in_dtype, signed char>()) {
            w_ir.SlideBias();
            w_hr.SlideBias();
            w_iz.SlideBias();
            w_hz.SlideBias();
            w_in.SlideBias();
            w_hn.SlideBias();
        }
    }

    // 順伝播
    inline void Forward(const Tensor<dtype, input_size>& input, Tensor<out_dtype, hidden_size>& hidden) {
        constexpr auto USE_AVX2 = true;
        if constexpr (USE_AVX2 && is_same<dtype, signed char>() && is_same<out_dtype, int>() && hidden_size % 32 == 0 && input_size % 32 == 0) {
            assert(((intptr_t)&hidden & 0b11111) == 0);
            alignas(32) static auto hidden_i8 = Tensor<dtype, hidden_size>();
            alignas(32) static auto r = Tensor<out_dtype, hidden_size>();
            alignas(32) static auto z = Tensor<out_dtype, hidden_size>();
            alignas(32) static auto tmp = Tensor<out_dtype, hidden_size>();

            hidden_i8 = hidden;

            w_ir.Forward(input, r);       // 7 -> 13
            w_hr.Forward(hidden_i8, tmp); // 7 -> 13
            r += tmp;                     // 13
            F::Sigmoid_(r);               // 13 -> 7

            w_hn.Forward(hidden_i8, tmp); // 7 -> 13
            r *= tmp;                     // 7 x 13 -> 20
            r >>= 7;                      // 20 -> 13
            w_in.Forward(input, tmp);     // 7 -> 13
            r += tmp;                     // 13
            F::Tanh_(r);                  // 13 -> 7

            w_iz.Forward(input, z);       // 7 -> 13
            w_hz.Forward(hidden_i8, tmp); // 7 -> 13
            z += tmp;                     // 13
            F::Sigmoid_(z);               // 13 -> 7

            hidden *= z;                  // 7 x 7 -> 14
            r *= z.SubtractFrom_(1 << 7); // 7 x 7 -> 14;
            hidden += r;                  // 14
            hidden >>= 7;                 // 14 -> 7
        } else if constexpr (USE_AVX2 && is_same<dtype, short>() && is_same<out_dtype, int>() && hidden_size % 16 == 0 && input_size % 16 == 0) {
            assert(((intptr_t)&hidden & 0b11111) == 0);
            alignas(32) static auto hidden_i16 = Tensor<dtype, hidden_size>();
            alignas(32) static auto r = Tensor<out_dtype, hidden_size>();
            alignas(32) static auto z = Tensor<out_dtype, hidden_size>();
            alignas(32) static auto tmp = Tensor<out_dtype, hidden_size>();
            hidden_i16 = hidden;

            w_ir.Forward(input, r);                             // 12 -> 13
            w_hr.Forward(hidden_i16, tmp);                      // 12 -> 13
            r += tmp;                                           // 13
            F::Sigmoid<decltype(r), decltype(r), 13, 12>(r, r); // 13 -> 12

            w_hn.Forward(hidden_i16, tmp);                   // 12 -> 13
            r *= tmp;                                        // 12 x 13 -> 25
            r >>= 12;                                        // 25 -> 13
            w_in.Forward(input, tmp);                        // 12 -> 13
            r += tmp;                                        // 13
            F::Tanh<decltype(r), decltype(r), 13, 12>(r, r); // 13 -> 12

            w_iz.Forward(input, z);                             // 12 -> 13
            w_hz.Forward(hidden_i16, tmp);                      // 12 -> 13
            z += tmp;                                           // 13
            F::Sigmoid<decltype(z), decltype(z), 13, 12>(z, z); // 13 -> 12

            hidden *= z;                   // 12 x 12 -> 24
            r *= z.SubtractFrom_(1 << 12); // 12 x 12 -> 24;
            hidden += r;                   // 24
            hidden >>= 12;                 // 24 -> 12
        } else if constexpr (is_same<dtype, float>() && is_same<out_dtype, float>()) {
            // 愚直
            static auto r = Tensor<out_dtype, hidden_size>();
            static auto z = Tensor<out_dtype, hidden_size>();
            static auto tmp = Tensor<out_dtype, hidden_size>();
            w_ir.Forward(input, r);
            w_hr.Forward(hidden, tmp);
            r += tmp;
            F::Sigmoid_(r);

            w_hn.Forward(hidden, tmp);
            r *= tmp;
            w_in.Forward(input, tmp);
            r += tmp;
            F::Tanh_(r);

            w_iz.Forward(input, z);
            w_hz.Forward(hidden, tmp);
            z += tmp;
            F::Sigmoid_(z);

            hidden *= z;
            r *= z.SubtractFrom_(1.0f);
            hidden += r;
        } else {
            assert(false);
        }
    }
};

template <int in_channels, int out_channels, int kernel_size, typename dtype = float, typename in_dtype = float, typename out_dtype = float>
struct alignas(32) Conv1d {
    // padding_mode="same" のみ対応

    // kernel_size は 1 と 3 のみ対応 (Winograd のアルゴリズムを使う都合)
    // kernel_size == 1 の場合は特殊化
    static_assert(kernel_size == 3);

    // float の重みのみ対応 (Winograd の係数に変換・量子化してから読み込まないといけなくて面倒なので)
    // float なら直接持ったほうが少ないメモリで済むけど、char だと変形後の重みを持ったほうが誤差が少なくなる どうすればいいんだ
    static_assert(is_same<dtype, float>() && is_same<in_dtype, float>() && is_same<out_dtype, float>());

    // パラメータ
    Tensor<dtype, out_channels, in_channels, kernel_size> weight; // 推論時には必要ない
    Tensor<dtype, out_channels, in_channels, 4> tw;
    Tensor<out_dtype, out_channels> bias;

    // 逐次生成用バッファ
    Tensor<dtype, 2, in_channels, 4> ti;
    Tensor<dtype, 2, out_channels, 4> to;
    Tensor<out_dtype, out_channels> out_buffer;
    unsigned clock;

    // コンストラクタ
    Conv1d() = default;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        if (fread(&weight, sizeof(dtype), weight.n_data, f) != weight.n_data)
            abort();
        if (fread(&bias, sizeof(out_dtype), bias.n_data, f) != bias.n_data)
            abort();
        CalcTransformedWeight();
    }

    // パラメータ読み込み
    void ReadParameters(const string& filename) {
        auto f = fopen(filename.c_str(), "rb");
        if (f == NULL) {
            cerr << filename << " を開けないよ" << endl;
            abort();
        }
        ReadParameters(f);
        fclose(f);
    }

    void CalcTransformedWeight() {
        for (int co = 0; co < out_channels; co++) {
            for (int ci = 0; ci < in_channels; ci++) {
                tw[co][ci][0] = weight[co][ci][0];
                tw[co][ci][1] = (weight[co][ci][0] + weight[co][ci][1] + weight[co][ci][2]) / 2;
                tw[co][ci][2] = (weight[co][ci][0] - weight[co][ci][1] + weight[co][ci][2]) / 2;
                tw[co][ci][3] = weight[co][ci][2];
            }
        }
    }

    // 順伝播
    template <int length> inline void Forward(const Tensor<dtype, length, in_channels>& input, Tensor<out_dtype, length, out_channels>& output) {
        // i: in
        // o: out
        // k: kernel
        // c: channel
        // t: transformed
        // w: weight
        // l: left
        // r: right
        constexpr auto pad = kernel_size / 2;
        if constexpr (false) {
            // 愚直
            for (int x = 0; x < length; x++) {
                for (int co = 0; co < out_channels; co++) {
                    output[x][co] = bias[co];
                }
                for (int kx = 0; kx < kernel_size; kx++) {
                    auto x_from = x + kx - pad;
                    if (x_from < 0 || x_from >= length)
                        continue;
                    for (int co = 0; co < out_channels; co++) {
                        for (int ci = 0; ci < in_channels; ci++) {
                            output[x][co] += input[x_from][ci] * weight[co][ci][kx];
                        }
                    }
                }
            }
        } else {
            // Winograd

            static_assert(length % 2 == 0);
            static_assert(kernel_size == 3);
            static auto ti = Tensor<dtype, 4>();
            static auto to = Tensor<dtype, 4, out_channels>();

            for (int x = 0; x < length; x += 2) {
                to.Fill((out_dtype)0);
                for (int ci = 0; ci < in_channels; ci++) {
                    ti[0] = (x != 0 ? input[x - 1][ci] : 0) - input[x + 1][ci];
                    ti[1] = input[x][ci] + input[x + 1][ci];
                    ti[2] = input[x + 1][ci] - input[x][ci];
                    ti[3] = (x + 2 != length ? input[x + 2][ci] : 0) - input[x][ci];
                    for (int co = 0; co < out_channels; co++) {
                        for (int xx = 0; xx < 4; xx++) {
                            to[xx][co] += ti[xx] * tw[co][ci][xx];
                        }
                    }
                }
                for (int co = 0; co < out_channels; co++) {
                    output[x][co] = bias[co] + to[0][co] + to[1][co] + to[2][co];
                    output[x + 1][co] = bias[co] + to[1][co] - to[2][co] + to[3][co];
                }
            }
        }
    }

    // 順伝播 (length 不定の場合) (出力は 2 フレーム (カーネル分 + Winograd 分) 遅れ)
    inline void Forward(const Tensor<dtype, in_channels>& input, Tensor<out_dtype, out_channels>& output) {
        auto& til = (clock & 2) ? ti[0] : ti[1];
        auto& tol = (clock & 2) ? to[0] : to[1];
        auto& tir = (clock & 2) ? ti[1] : ti[0];
        if ((clock & 1) == 0) {
            for (int ci = 0; ci < in_channels; ci++) {
                til[ci][0] -= input[ci];
                til[ci][1] += input[ci];
                til[ci][2] += input[ci];
                tir[ci][0] += input[ci];
            }
            output = out_buffer;
        } else {
            for (int ci = 0; ci < in_channels; ci++) {
                til[ci][3] += input[ci];
                tir[ci][1] += input[ci];
                tir[ci][2] -= input[ci];
                tir[ci][3] -= input[ci];
            }
            output = out_buffer = bias;
            for (int co = 0; co < out_channels; co++) {
                for (int ci = 0; ci < in_channels; ci++) {
                    for (int x = 0; x < 4; x++) {
                        tol[co][x] += til[ci][x] * tw[co][ci][x];
                    }
                }
                output[co] += tol[co][0] + tol[co][1] + tol[co][2];
                out_buffer[co] += tol[co][1] - tol[co][2] + tol[co][3];
            }
            til.Fill((dtype)0);
            tol.Fill((dtype)0);
        }
        clock++;
    }
};

template <int in_channels, int out_channels, typename dtype, typename in_dtype, typename out_dtype>
struct alignas(32) Conv1d<in_channels, out_channels, 1, dtype, in_dtype, out_dtype> {
    // 部分特殊化: kernel_size == 1 の場合
    constexpr static auto kernel_size = 1;
    using InTensorType = Tensor<in_dtype, in_channels>;
    using OutTensorType = Tensor<out_dtype, out_channels>;

    // パラメータ
    Linear<in_channels, out_channels, dtype, in_dtype, out_dtype> linear;

    // コンストラクタ
    Conv1d() = default;

    // パラメータ読み込み
    void ReadParameters(FILE* const f) { linear.ReadParameters(f); }

    // パラメータ読み込み
    void ReadParameters(const string& filename) {
        auto f = fopen(filename.c_str(), "rb");
        if (f == NULL) {
            cerr << filename << " を開けないよ" << endl;
            abort();
        }
        ReadParameters(f);
        fclose(f);
    }

    // 順伝播
    template <int length> inline void Forward(const Tensor<in_dtype, length, in_channels>& input, Tensor<out_dtype, length, out_channels>& output) {
        for (int x = 0; x < length; x++) {
            linear.Forward(input[x], output[x]);
        }
    }

    // 順伝播 (length 不定の場合)
    inline void Forward(const InTensorType& input, OutTensorType& output) { linear.Forward(input, output); }
};

template <int n_features> struct alignas(32) BatchNorm {
    // 1D, 2D 共通
    // float のみ対応

    using dtype = float;
    using in_dtype = float;

    constexpr static auto eps = 1e-5;
    Tensor<float, n_features> gamma;
    Tensor<float, n_features> beta;
    Tensor<float, n_features> running_mean;
    Tensor<float, n_features> running_var;
    Tensor<dtype, n_features> weight;
    Tensor<dtype, n_features> bias;

    // コンストラクタ
    BatchNorm() {
        weight.Fill(1.0f);
        bias.Fill(0.0f);
    }

    // パラメータ読み込み
    void ReadParameters(FILE* const f) {
        if (fread(&gamma, sizeof(dtype), n_features, f) != n_features)
            abort();
        if (fread(&beta, sizeof(dtype), n_features, f) != n_features)
            abort();
        if (fread(&running_mean, sizeof(dtype), n_features, f) != n_features)
            abort();
        if (fread(&running_var, sizeof(dtype), n_features, f) != n_features)
            abort();
        for (int c = 0; c < n_features; c++) {
            weight[c] = gamma[c] / sqrtf(running_var[c] + eps);
            bias[c] = beta[c] - weight[c] * running_mean[c];
        }
    }

    // パラメータ読み込み
    void ReadParameters(const string& filename) {
        auto f = fopen(filename.c_str(), "rb");
        if (f == NULL) {
            cerr << filename << " を開けないよ" << endl;
            abort();
        }
        ReadParameters(f);
        fclose(f);
    }

    // 順伝播
    // length 不定でも正しく動くはず
    template <int dim2, int dim3, int dim4> void Forward_(Tensor<in_dtype, n_features, dim2, dim3, dim4>& input) const {
        for (int channel = 0; channel < n_features; channel++) {
            input[channel] *= weight[channel];
            input[channel] += bias[channel];
        }
    }
};

template <int dim> struct alignas(32) ResBlock {
    Conv1d<dim, dim, 1> conv1, conv2;
    BatchNorm<dim> batchnorm1, batchnorm2;

    void ReadParameters(FILE* const f) {
        conv1.ReadParameters(f);
        batchnorm1.ReadParameters(f);
        conv2.ReadParameters(f);
        batchnorm2.ReadParameters(f);
    }

    // 順伝播 (length 不定の場合)
    void Forward_(Tensor<float, dim>& input) {
        static Tensor<float, dim> tmp1, tmp2;
        conv1.Forward(input, tmp1);
        batchnorm1.Forward_(tmp1);
        F::Relu_(tmp1);
        conv2.Forward(tmp1, tmp2);
        batchnorm2.Forward_(tmp2);
        input += tmp2;
        F::Relu_(input);
    }
};

} // namespace nn

#define rep(i, n) for (auto i = 0; (i) < (n); (i)++)
#define rep1(i, n) for (auto i = 1; (i) <= (n); (i)++)
#define rep3(i, s, n) for (auto i = (s); (i) < (n); (i)++)

//#define NDEBUG

#ifndef NDEBUG
#define VISUALIZE
#endif

#ifndef NDEBUG
#define ASSERT(expr, ...)                                                                                                                            \
    do {                                                                                                                                             \
        if (!(expr)) {                                                                                                                               \
            printf("%s(%d): Assertion failed.\n", __FILE__, __LINE__);                                                                               \
            printf(__VA_ARGS__);                                                                                                                     \
            abort();                                                                                                                                 \
        }                                                                                                                                            \
    } while (false)
#else
#define ASSERT(...)
#endif

#define ASSERT_RANGE(value, left, right) ASSERT((left <= value) && (value < right), "`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var)                                                                                                                                   \
    do {                                                                                                                                             \
        cerr << #var << '=' << var << endl;                                                                                                          \
    } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template <class T, class S> inline bool chmin(T& m, S q) {
    if (m > q) {
        m = q;
        return true;
    } else
        return false;
}

template <class T, class S> inline bool chmax(T& m, const S q) {
    if (m < q) {
        m = q;
        return true;
    } else
        return false;
}

// 乱数
struct Random {
    using ull = unsigned long long;
    unsigned seed;
    inline Random(const unsigned& seed_) : seed(seed_) { ASSERT(seed != 0u, "Seed should not be 0."); }
    const inline unsigned& next() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    }
    // (0.0, 1.0)
    inline double random() { return (double)next() * (1.0 / (double)0x100000000ull); }
    // [0, right)
    inline int randint(const int& right) { return (ull)next() * right >> 32; }
    // [left, right)
    inline int randint(const int& left, const int& right) { return ((ull)next() * (right - left) >> 32) + left; }
};

// 2 次元ベクトル
template <typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    T y, x;
    constexpr inline Vec2() = default;
    constexpr inline Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;            // コピー
    inline Vec2(Vec2&&) = default;                 // ムーブ
    inline Vec2& operator=(const Vec2&) = default; // 代入
    inline Vec2& operator=(Vec2&&) = default;      // ムーブ代入
    template <typename S> constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const { return Vec2(y + rhs.y, x + rhs.x); }
    inline Vec2 operator+(const T& rhs) const { return Vec2(y + rhs, x + rhs); }
    inline Vec2 operator-(const Vec2& rhs) const { return Vec2(y - rhs.y, x - rhs.x); }
    template <typename S> inline Vec2 operator*(const S& rhs) const { return Vec2(y * rhs, x * rhs); }
    inline Vec2 operator*(const Vec2& rhs) const { // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template <typename S> inline Vec2 operator/(const S& rhs) const {
        ASSERT(rhs != 0.0, "Zero division!");
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const { // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template <typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) {
        *this = (*this) * rhs;
        return *this;
    }
    inline Vec2& operator/=(const Vec2& rhs) {
        *this = (*this) / rhs;
        return *this;
    }
    inline bool operator!=(const Vec2& rhs) const { return x != rhs.x || y != rhs.y; }
    inline bool operator==(const Vec2& rhs) const { return x == rhs.x && y == rhs.y; }
    inline void rotate(const double& rad) { *this = rotated(rad); }
    inline Vec2<double> rotated(const double& rad) const { return (*this) * rotation(rad); }
    static inline Vec2<double> rotation(const double& rad) { return Vec2(sin(rad), cos(rad)); }
    static inline Vec2<double> rotation_deg(const double& deg) { return rotation(PI * deg / 180.0); }
    inline Vec2<double> rounded() const { return Vec2<double>(round(y), round(x)); }
    inline Vec2<double> inv() const { // x + yj とみなす
        const double norm_sq = l2_norm_square();
        ASSERT(norm_sq != 0.0, "Zero division!");
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const { return sqrt(x * x + y * y); }
    inline double l2_norm_square() const { return x * x + y * y; }
    inline T l1_norm() const { return std::abs(x) + std::abs(y); }
    inline double abs() const { return l2_norm(); }
    inline double phase() const { // [-PI, PI) のはず
        return atan2(y, x);
    }
    inline double phase_deg() const { // [-180, 180) のはず
        return phase() / PI * 180.0;
    }
};
template <typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) { return rhs * lhs; }
template <typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

// 2 次元配列
template <class T, int height, int width> struct Board {
    array<T, height * width> data;
    template <class Int> constexpr inline auto& operator[](const Vec2<Int>& p) { return data[width * p.y + p.x]; }
    template <class Int> constexpr inline const auto& operator[](const Vec2<Int>& p) const { return data[width * p.y + p.x]; }
    template <class Int> constexpr inline auto& operator[](const initializer_list<Int>& p) { return data[width * *p.begin() + *(p.begin() + 1)]; }
    template <class Int> constexpr inline const auto& operator[](const initializer_list<Int>& p) const {
        return data[width * *p.begin() + *(p.begin() + 1)];
    }
    constexpr inline void Fill(const T& fill_value) { fill(data.begin(), data.end(), fill_value); }
    void Print() const {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cout << data[width * y + x] << " \n"[x == width - 1];
            }
        }
    }
};

// スタック  // コンストラクタ呼ぶタイミングとかが考えられてなくて良くない
template <class T, int max_size> struct Stack {
    array<T, max_size> data;
    int right;
    inline Stack() : data(), right(0) {}
    inline Stack(const int n) : data(), right(0) { resize(n); }
    inline Stack(const int n, const T& val) : data(), right(0) { resize(n, val); }
    inline Stack(const initializer_list<T>& init) : data(), right(init.size()) {
        memcpy(&data[0], init.begin(), sizeof(T) * init.size());
    }                                                           // これ大丈夫か？
    inline Stack(const Stack& rhs) : data(), right(rhs.right) { // コピー
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
    }
    template <class S> inline Stack(const Stack<S, max_size>& rhs) : data(), right(rhs.right) {
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
    }
    template <class Container> Stack& operator=(const Container& rhs) {
        right = rhs.size();
        ASSERT(right <= max_size, "Too big container.");
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
        return *this;
    }
    Stack& operator=(Stack&&) = default;
    inline bool empty() const { return 0 == right; }
    inline void push(const T& value) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = value;
        right++;
    }
    inline T pop() {
        right--;
        ASSERT_RANGE(right, 0, max_size);
        return data[right];
    }
    const inline T& top() const { return data[right - 1]; }
    template <class... Args> inline void emplace(const Args&... args) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = T(args...);
        right++;
    }
    inline void clear() { right = 0; }
    inline void insert(const int& idx, const T& value) {
        ASSERT_RANGE(idx, 0, right + 1);
        ASSERT_RANGE(right, 0, max_size);
        int i = right;
        right++;
        while (i != idx) {
            data[i] = data[i - 1];
            i--;
        }
        data[idx] = value;
    }
    inline void del(const int& idx) {
        ASSERT_RANGE(idx, 0, right);
        right--;
        for (int i = idx; i < right; i++) {
            data[i] = data[i + 1];
        }
    }
    inline int index(const T& value) const {
        for (int i = 0; i < right; i++) {
            if (value == data[i])
                return i;
        }
        return -1;
    }
    inline void remove(const T& value) {
        int idx = index(value);
        ASSERT(idx != -1, "not contain the value.");
        del(idx);
    }
    inline void resize(const int& sz) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new (&data[right]) T();
        }
        right = sz;
    }
    inline void resize(const int& sz, const T& fill_value) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new (&data[right]) T(fill_value);
        }
        right = sz;
    }
    inline int size() const { return right; }
    inline T& operator[](const int n) {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline const T& operator[](const int n) const {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline T* begin() { return (T*)data.data(); }
    inline const T* begin() const { return (const T*)data.data(); }
    inline T* end() { return (T*)data.data() + right; }
    inline const T* end() const { return (const T*)data.data() + right; }
    inline T& front() {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    const inline T& front() const {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    inline T& back() {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }
    const inline T& back() const {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }
    inline bool contains(const T& value) const {
        for (const auto& dat : *this) {
            if (value == dat)
                return true;
        }
        return false;
    }
    inline vector<T> ToVector() { return vector<T>(begin(), end()); }
    inline void Print(ostream& os = cout) {
        for (int i = 0; i < right; i++) {
            os << data[i] << (i == right - 1 ? "" : " ");
        }
        os << endl;
    }
};

template <class T, int size = 0x100000, class KeyType = unsigned> struct MinimumHashMap {
    // ハッシュの値が size 以下
    array<T, size> data;
    Stack<int, size> used;
    constexpr static KeyType mask = size - 1;
    inline MinimumHashMap() {
        static_assert((size & size - 1) == 0, "not pow of 2");
        memset(&data[0], (unsigned char)-1, sizeof(data));
    }
    inline T& operator[](const KeyType& key) {
        if (data[key] == (T)-1)
            used.push(key);
        return data[key];
    }
    inline void clear() {
        for (const auto& key : used)
            data[key] = (T)-1;
        used.right = 0;
    }
};

// 時間 (秒)
inline double time() {
    return static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count()) * 1e-9;
}

template <typename T> struct Slice {
    T *left, *right;
    inline Slice(T* const& l, T* const& r) : left(l), right(r) {}
    inline T* begin() { return left; }
    inline const T* begin() const { return (const T*)left; }
    inline T* end() { return right; }
    inline const T* end() const { return (const T*)right; }
    inline int size() const { return distance(left, right); }
    inline T& operator[](const int& idx) { return left[idx]; }
    inline const T& operator[](const int& idx) const { return left[idx]; }
};

// ===========================================================================================================

enum struct Directions { U, D, L, R, NONE };
constexpr auto DIRECTION_VECS = array<Vec2<int>, 4>{Vec2<int>{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
enum struct PetType { NONE, COW, PIG, RABBIT, DOG, CAT };
using i8 = signed char;

namespace common {
int N; // ペットの数
int M; // 人間の数
auto pet_types = array<PetType, 20>();
auto pet_positions = array<Vec2<int>, 20>();
auto human_positions = array<Vec2<int>, 10>();

struct PetMove {
    Vec2<i8> position;
    char direction;
};
auto last_pet_moves = array<PetMove, 20>();
// auto pet_history = array<array<Directions, 300>, 20>();
auto current_turn = 0;
// auto global_features = nn::Tensor < fl

auto human_moves = array<char, 10>();
auto fence_board = Board<bool, 32, 32>();
auto human_count_board = Board<i8, 32, 32>();
auto pet_count_board = Board<i8, 32, 32>();

auto pet_moves = array<string, 20>();

auto elapsed_turn_after_dog_meets_human = array<int, 20>();
auto dog_targeting_probability = Board<double, 20, 10>();

auto legal_actions = array<short, 10>();

} // namespace common

namespace features {

constexpr auto N_GLOBAL_FEATURES = 25;
constexpr auto N_LOCAL_FEATURES = 31;
alignas(64) auto distances_from_each_human = array<Board<short, 32, 32>, 10>();
alignas(64) auto distances_from_each_pet = array<Board<short, 32, 32>, 20>();

// 0 -> 人
// 1 -> 柵
alignas(64) auto observation_local = nn::Tensor<float, N_LOCAL_FEATURES, 32, 32>();

} // namespace features

void Initialize() {
    using namespace common;
    cin >> N;
    rep(i, N) {
        int pt;
        cin >> pet_positions[i].y >> pet_positions[i].x >> pt;
        pet_count_board[pet_positions[i]]++;
        pet_types[i] = (PetType)pt;
    }
    cin >> M;
    rep(i, M) {
        cin >> human_positions[i].y >> human_positions[i].x;
        human_count_board[human_positions[i]]++;
    }
    rep(i, 32) {
        fence_board[{i, 0}] = true;
        fence_board[{i, 31}] = true;
        fence_board[{0, i}] = true;
        fence_board[{31, i}] = true;
        features::observation_local[1][i][0] = 1.0f;
        features::observation_local[1][i][31] = 1.0f;
        features::observation_local[1][0][i] = 1.0f;
        features::observation_local[1][31][i] = 1.0f;
    }
    dog_targeting_probability.Fill(1.0 / (double)M);
    rep(i, N) pet_moves[i] = "..";
}

void UpdateHuman() {
    // human_moves をセットした後に呼ぶ

    using common::fence_board;
    using common::human_count_board;
    using common::human_moves;
    using common::human_positions;
    using common::pet_count_board;
    using features::observation_local;

    // 仕切り設置処理
    auto set_fence = [&](const Vec2<int>& p) {
        if (human_count_board[p] != 0 || pet_count_board[p] != 0) {
            return false;
        } else {
            for (const auto& d : DIRECTION_VECS) {
                auto neighbor = p + d;
                if (pet_count_board[neighbor] != 0) {
                    return false;
                }
            }
        }
        fence_board[p] = true;
        return true;
    };

    // 先に通行不能化処理をやる
    rep(i, common::M) {
        const auto& v = human_positions[i];
        auto d = Directions::NONE;
        switch (human_moves[i]) {
        // 通行不能にする
        case 'u':
            d = Directions::U;
            break;
        case 'd':
            d = Directions::D;
            break;
        case 'l':
            d = Directions::L;
            break;
        case 'r':
            d = Directions::R;
            break;
        }
        if (d != Directions::NONE) {
            const auto u = v + DIRECTION_VECS[(int)d];
            const auto succeeded = set_fence(u);
            assert(succeeded);
            observation_local[1][u.y][u.x] = 1.0f;
        }
    }

    // 移動処理をする
    rep(i, common::M) {
        auto& v = human_positions[i];
        auto d = Directions::NONE;
        switch (human_moves[i]) {
        // 移動
        case 'U':
            d = Directions::U;
            break;
        case 'D':
            d = Directions::D;
            break;
        case 'L':
            d = Directions::L;
            break;
        case 'R':
            d = Directions::R;
            break;
        }
        if (d != Directions::NONE) {
            human_count_board[v]--;
            v += DIRECTION_VECS[(int)d];
            assert(!fence_board[v]);
            human_positions[i] = v;
            human_count_board[v]++;
        }
    }
}

void UpdatePets() {
    // pet_moves をセットした後に呼ぶ
    using common::current_turn;
    using common::fence_board;
    using common::pet_count_board;
    // using common::pet_history;
    using common::dog_targeting_probability;
    using common::elapsed_turn_after_dog_meets_human;
    using common::human_count_board;
    using common::human_positions;
    using common::last_pet_moves;
    using common::pet_moves;
    using common::pet_positions;
    using common::pet_types;
    rep(i, common::N) {
        const auto& pet_move = pet_moves[i];
        auto& v = pet_positions[i];
        pet_count_board[v]--;
        last_pet_moves[i] = {v, pet_move[0]};
        const auto is_dog = pet_types[i] == PetType::DOG;
        if (is_dog && pet_move.size() == 2) {
            // TODO: 犬関連について、孤立した犬の値が特徴量に混入していないか確認
            elapsed_turn_after_dog_meets_human[i]++;
            if (human_count_board[v] != 0) {
                elapsed_turn_after_dog_meets_human[i] = 1;
                // 人間と接したことによる確率の更新
                auto sum_proba = 0.0;
                rep(idx_human, common::M) {
                    if (v == human_positions[idx_human]) {
                        dog_targeting_probability[{i, idx_human}] = 1e-20;
                    }
                    sum_proba += dog_targeting_probability[{i, idx_human}];
                }
                rep(idx_human, common::M) { dog_targeting_probability[{i, idx_human}] *= (1.0 / sum_proba); }
            }
            // 人間と切り離されたことによる確率の更新
            {
                auto sum_proba = 0.0;
                rep(idx_human, common::M) {
                    if (features::distances_from_each_human[idx_human][v] == 999) {
                        dog_targeting_probability[{i, idx_human}] = 1e-20;
                    }
                    sum_proba += dog_targeting_probability[{i, idx_human}];
                }
                rep(idx_human, common::M) { dog_targeting_probability[{i, idx_human}] *= (1.0 / sum_proba); }
            }
        }
        rep(idx_mv, (int)pet_move.size()) {
            const auto& mv = pet_move[idx_mv];
            {

                Directions d;
                switch (mv) {
                case 'U':
                    d = Directions::U;
                    break;
                case 'D':
                    d = Directions::D;
                    break;
                case 'L':
                    d = Directions::L;
                    break;
                case 'R':
                    d = Directions::R;
                    break;
                }
                const auto u = v + DIRECTION_VECS[(int)d];
                assert(!fence_board[u]);

                // 1 回目の移動による確率の更新  // distances_from_each_human は先に計算しないといけない
                if (is_dog && idx_mv == 0 && pet_move.size() == 2) {
                    auto sum_proba = 0.0;
                    rep(idx_human, common::M) {
                        auto min_dist = 999;
                        auto min_cnt = 1;
                        for (const auto& dd : DIRECTION_VECS) {
                            if (min_dist > features::distances_from_each_human[idx_human][v + dd]) {
                                min_dist = features::distances_from_each_human[idx_human][v + dd];
                                min_cnt = 1;
                            } else if (min_dist == features::distances_from_each_human[idx_human][v + dd]) {
                                min_cnt++;
                            }
                        }
                        if (features::distances_from_each_human[idx_human][u] == min_dist) {
                            dog_targeting_probability[{i, idx_human}] /= (double)min_cnt;
                        } else {
                            dog_targeting_probability[{i, idx_human}] = 1e-20;
                        }
                        sum_proba += dog_targeting_probability[{i, idx_human}];
                    }
                    rep(idx_human, common::M) { dog_targeting_probability[{i, idx_human}] *= (1.0 / sum_proba); }
                }
                v = u;
            }
            if (is_dog && pet_move.size() == 2) {
                if (human_count_board[v] != 0) {
                    elapsed_turn_after_dog_meets_human[i] = 0;
                    // 人間と接したことによる確率の更新
                    auto sum_proba = 0.0;
                    rep(idx_human, common::M) {
                        if (v == human_positions[idx_human]) {
                            dog_targeting_probability[{i, idx_human}] = 1e-20;
                        }
                        sum_proba += dog_targeting_probability[{i, idx_human}];
                    }
                    rep(idx_human, common::M) { dog_targeting_probability[{i, idx_human}] *= (1.0 / sum_proba); }
                }
            }
        }
        pet_count_board[v]++;
        if (is_dog && pet_move.size() != 2) {
            elapsed_turn_after_dog_meets_human[i] = 0;
        }

        // pet_history[i][current_turn] = ; // まあ、いらんか…
    }
}

namespace features {

// global features
auto turn = 0;
auto n_pet = 0;
auto n_human = 0;
auto log_remaining_turns = 0.0;
auto human_y_mean = 0.0;
auto human_x_mean = 0.0;
auto human_y_std = 0.0;
auto human_x_std = 0.0;
auto human_yx_corr = 0.0;
auto pet_y_mean = 0.0;
auto pet_x_mean = 0.0;
auto pet_y_std = 0.0;
auto pet_x_std = 0.0;
auto pet_yx_corr = 0.0;
auto n_fences = 0;
auto n_fences_per_turn = 0.0;
auto max_area = 0;
auto sum_n_remaining_pet = 0; // これは使わない
auto n_remaining_pet_each_type = array<int, 5>();
auto n_surviving_human = 0;
auto diff_odd_even_dog_cat = 0;
auto max_area_human = 0;

// local features
// auto human_counts = Board<i8, 32, 32>();
// auto fence = Board<int, 32, 32>();
auto pet_counts = array<Board<int, 32, 32>, 5>();

alignas(64) auto min_distance_from_human = Board<short, 32, 32>();
alignas(64) auto min_distance_from_pet = Board<short, 32, 32>();
alignas(64) auto min_distance_pet_type = Board<PetType, 32, 32>();

auto area_each_human = array<int, 10>();
auto area_each_pet = array<int, 20>();

auto observation_global = nn::Tensor<float, N_GLOBAL_FEATURES>();
} // namespace features

namespace lowlink {

auto order = Board<short, 32, 32>();
auto low = Board<short, 32, 32>();
auto current_order = 0;
auto articulation_board = Board<bool, 32, 32>();
auto articulation_fence_board = Board<bool, 32, 32>();
struct ArticulationPoint {
    Vec2<i8> position;
    Vec2<i8> widest_dir;
    short area; // その関節点を封鎖することで失う面積、人、ペット (関節点を含めない)
    i8 n_human;
    i8 n_pets;
};
auto articulation_points = array<ArticulationPoint, 900>();
auto n_articulation_points = 0;
auto non_articulation_points = array<Vec2<i8>, 900>();
auto n_non_articulation_points = 0;
auto subtree_size = Board<short, 32, 32>();
auto n_pets_in_subtree = Board<i8, 32, 32>();
auto n_human_in_subtree = Board<i8, 32, 32>();

struct Edge {
    Vec2<i8> from, to;
};
// auto edges = array<Edge, 900>(); // DFS 木の辺
// auto n_edges = 0;

void Dfs(const Vec2<i8>& v, const Vec2<i8>& p) {
    order[v] = low[v] = ++current_order;
    auto is_articulation = false;
    auto n_children = (short)0;
    auto neighboring_widest_2cc = Vec2<i8>{-1, -1};
    auto neighboring_widest_2cc_area = (short)0;
    auto loss_2cc_area = (short)0;
    auto n_victim_human = (i8)0;
    auto n_capturable_pets = (i8)0;
    for (const auto& d : DIRECTION_VECS) {
        const auto u = v + d;
        if (common::fence_board[u])
            continue;
        if (order[u] == -1) {
            n_children++;
            Dfs(u, v);
            chmin(low[v], low[u]);
            if (order[v] <= low[u]) {
                if (p.y != -1) {
                    is_articulation = true;
                }
                if (chmax(neighboring_widest_2cc_area, subtree_size[u])) {
                    neighboring_widest_2cc = u;
                }
                loss_2cc_area += subtree_size[u];
                n_victim_human += n_human_in_subtree[u];
                n_capturable_pets += n_pets_in_subtree[u];
            }
            // edges[n_edges++] = {v, u};
            subtree_size[v] += subtree_size[u];
            n_human_in_subtree[v] += n_human_in_subtree[u];
            n_pets_in_subtree[v] += n_pets_in_subtree[u];
        } else if (u != p) {
            chmin(low[v], order[u]);
        }
    }
    is_articulation |= p.y == -1 && n_children >= 2;
    if (is_articulation) {
        if (p.y != -1 && neighboring_widest_2cc_area < features::max_area - loss_2cc_area - 1) {
            // 親を含む 2-連結成分が最大  p.y != -1 の条件は要らないと思うけど念の為
            articulation_points[n_articulation_points++] = {v, p - v, loss_2cc_area, n_victim_human, n_capturable_pets};
        } else {
            // 子が最大
            articulation_points[n_articulation_points++] = {
                v, neighboring_widest_2cc - v, (short)(features::max_area - neighboring_widest_2cc_area - 1),
                (i8)(features::n_surviving_human - n_human_in_subtree[neighboring_widest_2cc] - common::human_count_board[v]),
                (i8)(features::sum_n_remaining_pet - n_pets_in_subtree[neighboring_widest_2cc] - common::pet_count_board[v])};
        }
    } else {
        non_articulation_points[n_non_articulation_points++] = v;
    }
    n_human_in_subtree[v] += common::human_count_board[v];
    n_pets_in_subtree[v] += common::pet_count_board[v];
}

void Compute(const Vec2<i8>& start) {
    order.Fill(-1);
    current_order = 0;
    articulation_board.Fill(false);
    articulation_fence_board = common::fence_board;
    n_articulation_points = 0;
    n_non_articulation_points = 0;
    subtree_size.Fill(1);
    n_pets_in_subtree.Fill(0);
    n_human_in_subtree.Fill(0);
    Dfs(start, {-1, -1});
    // cout << "# subtree_size=" << subtree_size[start] << endl;
    assert(subtree_size[start] == features::max_area);
    assert(n_pets_in_subtree[start] == features::sum_n_remaining_pet);
    assert(n_human_in_subtree[start] == features::n_surviving_human);
}

} // namespace lowlink

void PreComputeFeatures() {
    // BFS
    namespace f = features;
    auto bfs = [](const auto& board, const auto& start, auto& distances) {
        distances.Fill(999);
        auto q = array<Vec2<int>, 900>();
        auto ql = &q[0];
        auto qr = ql;
        *qr = start;
        qr++;
        distances[start] = 0;
        do {
            const auto& v = *ql;
            ql++;
            for (const auto& d : DIRECTION_VECS) {
                const auto u = v + d;
                if (!board[u] && distances[u] == 999) {
                    distances[u] = distances[v] + 1;
                    *qr = u;
                    qr++;
                }
            }
        } while (ql != qr);
        return distance(&q[0], ql); // 面積を返す
    };
    rep(i, common::M) { f::area_each_human[i] = bfs(common::fence_board, common::human_positions[i], f::distances_from_each_human[i]); }
    rep(i, common::N) { f::area_each_pet[i] = bfs(common::fence_board, common::pet_positions[i], f::distances_from_each_pet[i]); }

    f::max_area = 0;
    rep(i, common::M) {
        if (chmax(f::max_area, f::area_each_human[i])) {
            f::max_area_human = i;
        }
    }
}

void ExtractFeatures() {
    // 特徴量抽出 (observation 作成)
    namespace f = features;
    auto& g = f::observation_global;
    auto& l = f::observation_local;
    auto idx_g = 0, idx_l = 0;

    using f::max_area_human;

    // --- global ---
    {

        // ターン数
        f::turn = common::current_turn;
        g[idx_g++] = f::turn * (1.0 / 300.0);

        // 全ペットの数、人の数
        f::n_pet = common::N;
        f::n_human = common::M;
        g[idx_g++] = (f::n_pet - 10) * (1.0 / 10.0);
        g[idx_g++] = (f::n_human - 5) * (1.0 / 5.0);

        // log1p(残りターン数)
        f::log_remaining_turns = log(300 - common::current_turn);
        g[idx_g++] = f::log_remaining_turns / log(300.0);

        // 人の位置
        auto sum_human_positions = Vec2<int>();
        auto sum_squared_human_positions = Vec2<int>();
        auto sum_human_yx = 0;
        rep(i, common::M) {
            const auto& p = common::human_positions[i];
            sum_human_positions += p;
            sum_squared_human_positions.y += p.y * p.y;
            sum_squared_human_positions.x += p.x * p.x;
            sum_human_yx += p.y * p.x;
        }
        f::human_y_mean = (double)sum_human_positions.y / (double)common::M;
        f::human_x_mean = (double)sum_human_positions.x / (double)common::M;
        f::human_y_std = sqrt((double)sum_squared_human_positions.y / (double)common::M - f::human_y_mean * f::human_y_mean);
        f::human_x_std = sqrt((double)sum_squared_human_positions.x / (double)common::M - f::human_x_mean * f::human_x_mean);
        f::human_yx_corr = ((double)sum_human_yx / (double)common::M - f::human_y_mean * f::human_x_mean) / (f::human_y_std * f::human_x_std);
        g[idx_g++] = (f::human_y_mean - 15.5) * (1.0 / 14.5);
        g[idx_g++] = (f::human_x_mean - 15.5) * (1.0 / 14.5);
        g[idx_g++] = f::human_y_std * (1.0 / 14.5);
        g[idx_g++] = f::human_x_std * (1.0 / 14.5);
        g[idx_g++] = f::human_yx_corr;

        // ペットの位置
        auto sum_pet_positions = Vec2<int>();
        auto sum_squared_pet_positions = Vec2<int>();
        auto sum_pet_yx = 0;
        rep(i, common::M) {
            const auto& p = common::pet_positions[i];
            sum_pet_positions += p;
            sum_squared_pet_positions.y += p.y * p.y;
            sum_squared_pet_positions.x += p.x * p.x;
            sum_pet_yx += p.y * p.x;
        }
        f::pet_y_mean = (double)sum_pet_positions.y / (double)common::N;
        f::pet_x_mean = (double)sum_pet_positions.x / (double)common::N;
        f::pet_y_std = sqrt((double)sum_squared_pet_positions.y / (double)common::N - f::pet_y_mean * f::pet_y_mean);
        f::pet_x_std = sqrt((double)sum_squared_pet_positions.x / (double)common::N - f::pet_x_mean * f::pet_x_mean);
        f::pet_yx_corr = ((double)sum_pet_yx / (double)common::N - f::pet_y_mean * f::pet_x_mean) / (f::pet_y_std * f::pet_x_std);
        g[idx_g++] = (f::pet_y_mean - 15.5) * (1.0 / 14.5);
        g[idx_g++] = (f::pet_x_mean - 15.5) * (1.0 / 14.5);
        g[idx_g++] = f::pet_y_std * (1.0 / 14.5);
        g[idx_g++] = f::pet_x_std * (1.0 / 14.5);
        g[idx_g++] = f::pet_yx_corr;

        // 柵の数
        f::n_fences = 0;
        rep1(y, 30) rep1(x, 30) f::n_fences += common::fence_board[{y, x}];
        g[idx_g++] = f::n_fences * (1.0 / 100.0);

        // 柵の数 / 経過ターン
        f::n_fences_per_turn = (double)f::n_fences / ((double)f::turn + 1e-14);
        g[idx_g++] = min(f::n_fences_per_turn, 3.0);

        // 最大面積
        g[idx_g++] = f::max_area * (1.0 / 900.0);

        // 各ペットの残り数 (最大エリアの人と違うところに居たら捕まえたと考える)
        f::sum_n_remaining_pet = 0;
        fill(f::n_remaining_pet_each_type.begin(), f::n_remaining_pet_each_type.end(), 0);
        rep(i, common::N) {
            const auto& pet_position = common::pet_positions[i];
            const auto& pet_type = common::pet_types[i];
            if (f::distances_from_each_human[max_area_human][pet_position] != 999) {
                f::n_remaining_pet_each_type[(int)pet_type - 1]++;
                f::sum_n_remaining_pet++;
            }
        }
        rep(pet_type, 5) { g[idx_g++] = f::n_remaining_pet_each_type[pet_type] * 0.25; }

        // 何人が最大面積のところに居るか
        f::n_surviving_human = 0;
        rep(i, common::M) {
            const auto& human_position = common::human_positions[i];
            f::n_surviving_human += f::distances_from_each_human[max_area_human][human_position] != 999;
        }
        g[idx_g++] = f::n_surviving_human * 0.1;

        // x + y が偶数の位置の犬猫の数と奇数の位置の犬猫の数の差
        f::diff_odd_even_dog_cat = 0;
        rep(i, common::N) {
            if ((int)common::pet_types[i] >= (int)PetType::DOG) {
                const auto& pos = common::pet_positions[i];
                if ((pos.y + pos.x) % 2) {
                    f::diff_odd_even_dog_cat++;
                } else {
                    f::diff_odd_even_dog_cat--;
                }
            }
        }
        g[idx_g++] = f::diff_odd_even_dog_cat * 0.1;

        // 2-連結成分の数
        lowlink::Compute(common::human_positions[max_area_human]);
        g[idx_g++] = lowlink::n_articulation_points * 0.01;

        assert(idx_g == f::N_GLOBAL_FEATURES);
    }

    // --- local ---
    {
        // 人  1
        l[idx_l].Fill(0.0f);
        rep(i, common::M) {
            const auto& p = common::human_positions[i];
            l[idx_l][p.y][p.x] += 1.0f;
        }
        idx_l++;

        // 仕切りは設置時に処理済み  1
        idx_l++;

        // x + y の偶奇  1
        if (common::current_turn == 0) {
            rep(y, 32) rep(x, 32) l[idx_l][y][x] = ((y + x) % 2) * 2 - 1;
        }
        idx_l++;

        // 最も近いペットからの道のり、種類  6
        // これキャッシュが衝突しそうだなあ…
        rep3(i, idx_l, idx_l + 6) { l[i].Fill(0.0f); }
        f::min_distance_from_human.Fill(999);
        f::min_distance_from_pet.Fill(999);
        f::min_distance_pet_type.Fill(PetType::COW);
        rep(i, common::N) {
            const auto& distance_board = f::distances_from_each_pet[i];
            rep(y, 32) rep(x, 32) {
                if (chmin(f::min_distance_from_pet[{y, x}], distance_board[{y, x}])) {
                    f::min_distance_pet_type[{y, x}] = common::pet_types[i];
                }
            }
        }
        rep(y, 32) rep(x, 32) {
            if (f::min_distance_from_pet[{y, x}] <= 50) {
                l[idx_l][y][x] = f::min_distance_from_pet[{y, x}] * 0.04;
            } else {
                l[idx_l][y][x] = (min((short)200, f::min_distance_from_pet[{y, x}]) - 50) * 0.01 + 2.0;
            }
            if (f::min_distance_from_pet[{y, x}] != 999) {
                l[idx_l + (int)f::min_distance_pet_type[{y, x}]][y][x]++;
            }
        }
        idx_l += 6;

        // 最も近い人からの道のり  1
        rep(i, common::M) {
            const auto& distance_board = f::distances_from_each_human[i];
            rep(y, 32) rep(x, 32) chmin(f::min_distance_from_human[{y, x}], distance_board[{y, x}]);
        }
        rep(y, 32) rep(x, 32) {
            if (f::min_distance_from_human[{y, x}] <= 50) {
                l[idx_l][y][x] = f::min_distance_from_human[{y, x}] * 0.04;
            } else {
                l[idx_l][y][x] = (min((short)200, f::min_distance_from_human[{y, x}]) - 50) * 0.01 + 2.0;
            }
        }
        idx_l++;

        // 犬猫の時間減衰移動痕 x/y  4
        // 前のターンの位置が必要
        if (common::current_turn != 0) {
            l[idx_l] *= 0.85;
            l[idx_l + 1] *= 0.85;
            l[idx_l + 2] *= 0.85;
            l[idx_l + 3] *= 0.85;
            rep(i, common::N) {
                auto pt = (int)common::pet_types[i];
                if (pt < (int)PetType::DOG)
                    continue;
                pt -= (int)PetType::DOG;
                auto& last_pos = common::last_pet_moves[i].position;
                switch (common::last_pet_moves[i].direction) {
                case 'U':
                    l[idx_l + pt * 2][last_pos.y][last_pos.x]--;
                    break;
                case 'D':
                    l[idx_l + pt * 2][last_pos.y][last_pos.x]++;
                    break;
                case 'L':
                    l[idx_l + pt * 2 + 1][last_pos.y][last_pos.x]--;
                    break;
                case 'R':
                    l[idx_l + pt * 2 + 1][last_pos.y][last_pos.x]++;
                    break;
                }
            }
        }
        idx_l += 4;

        // 犬なら、最後に人と重なってからの経過ターン  1
        // 犬なら、移動確率  4
        l[idx_l].Fill(0);
        l[idx_l + 1].Fill(0);
        l[idx_l + 2].Fill(0);
        l[idx_l + 3].Fill(0);
        l[idx_l + 4].Fill(0);
        rep(i, common::N) {
            if (common::pet_types[i] == PetType::DOG && common::pet_moves[i].size() == 2) {
                const auto& pos = common::pet_positions[i];
                l[idx_l][pos.y][pos.x] = common::elapsed_turn_after_dog_meets_human[i] * 0.04;
                // 確率
                rep(idx_human, common::M) {
                    auto min_dist = 999;
                    auto min_cnt = 1;
                    for (const auto& d : DIRECTION_VECS) {
                        if (min_dist > features::distances_from_each_human[idx_human][pos + d]) {
                            min_dist = features::distances_from_each_human[idx_human][pos + d];
                            min_cnt = 1;
                        } else if (min_dist == features::distances_from_each_human[idx_human][pos + d]) {
                            min_cnt++;
                        }
                    }
                    rep(idx_d, 4) {
                        const auto& d = DIRECTION_VECS[idx_d];
                        if (min_dist == features::distances_from_each_human[idx_human][pos + d]) {
                            l[idx_l + 1 + idx_d] += common::dog_targeting_probability[{i, idx_human}] / (double)min_cnt;
                        }
                    }
                }
            }
        }
        idx_l += 5;

        // 人なら、犬から逃れる方角、最も狙われている確率が高い犬からの道のり  5
        l[idx_l].Fill(0);
        l[idx_l + 1].Fill(0);
        l[idx_l + 2].Fill(0);
        l[idx_l + 3].Fill(0);
        l[idx_l + 4].Fill(0);
        rep(idx_human, common::M) {
            auto max_proba = 0.0;
            auto max_proba_distance = 0;
            const auto& human_pos = common::human_positions[idx_human];
            rep(i, common::N) {
                if (common::pet_types[i] == PetType::DOG && common::pet_moves[i].size() == 2) {
                    auto best_dir = -1;
                    auto max_dist = 0;
                    const auto& v = common::pet_positions[i];
                    rep(i, 4) {
                        const auto u = v + DIRECTION_VECS[i];
                        if (f::distances_from_each_pet[i][u] == 999)
                            continue;
                        if (chmax(max_dist, f::distances_from_each_pet[i][u])) {
                            best_dir = i;
                        }
                    }
                    if (best_dir != -1) { // この条件いる？
                        l[idx_l + best_dir][human_pos.y][human_pos.x] += common::dog_targeting_probability[{i, idx_human}];
                        if (chmax(max_proba, common::dog_targeting_probability[{i, idx_human}])) {
                            max_proba_distance = f::distances_from_each_pet[i][v];
                        }
                    }
                }
            }
            l[idx_l + 4][human_pos.y][human_pos.x] = min(max_proba_distance, 50) * 0.04;
        }
        idx_l += 5;

        // 2-連結成分内の面積、人の数、ペットの数  3 // これいらんくないか？

        // 関節点に対して、そこを封鎖したとき最大エリアは何マス縮むか、最大エリア以外に何人/何頭いるか、封鎖した後の最大エリアはどっちか  5
        // 3 つ以上に分かれる
        l[idx_l].Fill(0);
        l[idx_l + 1].Fill(0);
        l[idx_l + 2].Fill(0);
        l[idx_l + 3].Fill(0);
        l[idx_l + 4].Fill(0);
        rep(i, lowlink::n_articulation_points) {
            const auto& ap = lowlink::articulation_points[i];
            l[idx_l][ap.position.y][ap.position.x] = ap.area * 0.02;
            l[idx_l + 1][ap.position.y][ap.position.x] = ap.n_human * 0.2;
            l[idx_l + 2][ap.position.y][ap.position.x] = ap.n_pets * 0.1;
            l[idx_l + 3][ap.position.y][ap.position.x] = ap.widest_dir.y;
            l[idx_l + 4][ap.position.y][ap.position.x] = ap.widest_dir.x;
        }
        idx_l += 5;

        // 1 手で封鎖できる門をすべて封鎖した後の連結成分内の面積、人の数、ペットの数、封鎖した門の数  4  // これもいらない？

        // k 移動後のペット存在確率？

        // そこは最大エリアか  1
        l[idx_l].Fill(0);
        rep(y, 32) rep(x, 32) { l[idx_l][y][x] = f::distances_from_each_human[max_area_human][{y, x}] != 999; }
        idx_l++;

        // 人に対して、そこの面積
        l[idx_l].Fill(0);
        rep(i, common::M) {
            const auto& v = common::human_positions[i];
            l[idx_l][v.y][v.x] = features::area_each_human[i] * (1.0 / 900.0);
        }
        idx_l++;

        assert(idx_l == f::N_LOCAL_FEATURES);
    }
}

void PrintFeatures() {
    // TODO
}

void ComputeLegalActions() {
    using common::legal_actions;
    fill(legal_actions.begin(), legal_actions.end(), 1 << 8);
    rep(i, common::M) {
        const auto& v = common::human_positions[i];
        rep(mv, 4) {
            const auto u = v + DIRECTION_VECS[mv];
            if (common::fence_board[u])
                continue;
            legal_actions[i] |= 1 << (mv + 4);
            if (common::pet_count_board[u] || common::human_count_board[u])
                continue;
            rep(mv2, 4) {
                if (common::pet_count_board[u + DIRECTION_VECS[mv2]])
                    goto break_continue;
            }
            legal_actions[i] |= 1 << mv;
        break_continue:;
        }
    }
}

namespace rl {

auto log_reward_ratio = 0.2;
auto reward = array<float, 10>();
auto cumulative_linear_reward = array<float, 10>();
auto cumulative_log_reward = array<float, 10>();
auto linear_outcome = array<float, 10>();
auto log_outcome = array<float, 10>();
auto outcome = array<float, 10>();
auto reward_coef = 5.0;

} // namespace rl

void ComputeReward() {
    // 各人に対して、捕まり率みたいなものを使って計算した点数を出力
    using rl::cumulative_linear_reward;
    using rl::cumulative_log_reward;
    using rl::log_reward_ratio;
    using rl::reward;
    using rl::reward_coef;

    auto capturability_point = array<float, 20>();
    // ペットごとにシミュレートしようかと思ったけど面倒だから全部ランダムでいいや…
    rep(i, common::N) {
        auto dp = Board<double, 32, 32>();
        const auto& v = common::pet_positions[i];
        dp[v] = 1.0;
        rep(t, 10) {
            auto new_dp = Board<double, 32, 32>();
            rep3(y, 1, 31) rep3(x, 1, 31) {
                const auto v = Vec2{y, x};
                auto n_open = 1e-100;
                for (const auto& d : DIRECTION_VECS) {
                    if (!common::fence_board[v + d]) {
                        n_open++;
                    }
                }
                for (const auto& d : DIRECTION_VECS) {
                    if (!common::fence_board[v + d]) {
                        new_dp[v + d] += dp[v] / n_open;
                    }
                }
            }
            dp = new_dp;
        }
        auto sum = Vec2{0.0, 0.0};
        auto sum_sq = Vec2{0.0, 0.0};
        rep3(y, 1, 31) rep3(x, 1, 31) {
            sum += dp[{y, x}] * Vec2<double>{y, x};
            sum_sq += dp[{y, x}] * Vec2<double>{y * y, x * x};
        }
        const auto var = sum_sq - Vec2{sum.y * sum.y, sum.x * sum.x};
        const auto std = sqrt((double)(var.y + var.x));
        constexpr auto MAX_STD = 3.1622776601683795; // sqrt(10) // これより大きくなることがあるみたいだけどよくわからん…
        capturability_point[i] = 1.0 - std * (1.0 / MAX_STD);
        // cout << "#capturability_point[i]=" << capturability_point[i] << endl;
    }

    auto new_cumulative_linear_reward = array<float, 10>();
    auto new_cumulative_log_reward = array<float, 10>();
    cout << "#reward=";
    rep(idx_human, common::M) {
        auto n = (double)common::N;
        reward[idx_human] = 0.0;
        rep(i, common::N) {
            if (features::distances_from_each_human[idx_human][common::pet_positions[i]] == 999) {
                n--;
            } else {
                n -= capturability_point[i];
            }
        }
        const auto& area = features::area_each_human[idx_human];
        new_cumulative_linear_reward[idx_human] = (1.0 / 900.0) * area * exp2(-n);
        new_cumulative_log_reward[idx_human] = (log2((1.0 / 900.0) * area) - n) / (double)common::N + 1.0;

        // あああああああああ
        if (features::distances_from_each_human[features::max_area_human][common::human_positions[idx_human]] != 999) {
            reward[idx_human] += 0.1;
        }
        for (const auto& d : DIRECTION_VECS) {
            if (!common::fence_board[common::human_positions[idx_human] + d])
                goto ok;
        }
        new_cumulative_linear_reward[idx_human] -= 2.0;
    ok:;

        reward[idx_human] += (new_cumulative_linear_reward[idx_human] - cumulative_linear_reward[idx_human]) * (1.0 - rl::log_reward_ratio) +
                             (new_cumulative_log_reward[idx_human] - cumulative_log_reward[idx_human]) * rl::log_reward_ratio;
        reward[idx_human] *= reward_coef;
        cout << reward[idx_human] << ",";
    }
    cout << endl;
    cumulative_linear_reward = new_cumulative_linear_reward;
    cumulative_log_reward = new_cumulative_log_reward;
}

void ComputeOutcome() {
    using rl::linear_outcome;
    using rl::log_outcome;
    using rl::log_reward_ratio;
    using rl::outcome;
    // 各人に対して点数を出力

    rep(idx_human, common::M) {
        auto n = (double)common::N;
        rep(i, common::N) {
            if (features::distances_from_each_human[idx_human][common::pet_positions[i]] == 999) {
                n--;
            }
        }
        const auto& area = features::area_each_human[idx_human];
        linear_outcome[idx_human] = (1.0 / 900.0) * area * exp2(-n);
        log_outcome[idx_human] = (log2((1.0 / 900.0) * area) - n) / (double)common::N + 1.0;
        outcome[idx_human] = linear_outcome[idx_human] * (1.0 - rl::log_reward_ratio) + log_outcome[idx_human] * rl::log_reward_ratio;
    }
}

void Predict() {
    // NN での予測と、予測値からの行動決定 (禁止操作の除外など)
    using common::human_moves;
    using features::observation_global;
    using features::observation_local;
    // TODO

    rep(i, common::M) { human_moves[i] = '.'; }
}

void Interact() {
    // 毎ターンの入出力
    using common::human_moves;
    using common::pet_moves;

    rep(i, common::M) { cout << human_moves[i]; }
    cout << endl;

    rep(i, common::N) { cin >> pet_moves[i]; }
}

void Solve() {
    Initialize();
    PreComputeFeatures();
    rep(_, 300) {
        ExtractFeatures();
        Predict();
        UpdateHuman();
        Interact();
        PreComputeFeatures();
        UpdatePets();
        common::current_turn++;
    }
}

#ifndef SKIP_MAIN
int main() {
    // std::this_thread::sleep_for(std::chrono::seconds(10));
    Solve();
}
#endif

#ifdef __clang__
#pragma clang attribute pop
#endif
