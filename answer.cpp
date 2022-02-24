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
auto next_human_positions = array<Vec2<int>, 10>();

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

constexpr auto starts_left = array<Vec2<i8>, 9>{Vec2<i8>{2, 1}, {5, 1}, {9, 2}, {12, 1}, {15, 1}, {19, 2}, {22, 1}, {25, 1}, {29, 2}};
constexpr auto starts_right = array<Vec2<i8>, 9>{Vec2<i8>{2, 29}, {5, 30}, {9, 30}, {12, 29}, {15, 30}, {19, 30}, {22, 29}, {25, 30}, {29, 30}};
auto distance_from_starts_left = array<Board<short, 32, 32>, 9>();
auto distance_from_starts_right = array<Board<short, 32, 32>, 9>();

} // namespace common

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
                v = u;
            }
        }
        pet_count_board[v]++;

        // pet_history[i][current_turn] = ; // まあ、いらんか…
    }
}

namespace features {
auto distances_from_each_human = array<Board<short, 32, 32>, 10>();
auto distances_from_each_pet = array<Board<short, 32, 32>, 20>();

auto area_each_human = array<int, 10>();
auto area_each_pet = array<int, 20>();
} // namespace features

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

void PreComputeFeatures() {
    // BFS
    namespace f = features;
    rep(i, common::M) { f::area_each_human[i] = bfs(common::fence_board, common::human_positions[i], f::distances_from_each_human[i]); }
    rep(i, common::N) { f::area_each_pet[i] = bfs(common::fence_board, common::pet_positions[i], f::distances_from_each_pet[i]); }
    rep(i, 9) {
        bfs(common::fence_board, common::starts_left[i], common::distance_from_starts_left[i]);
        bfs(common::fence_board, common::starts_right[i], common::distance_from_starts_right[i]);
    }
}

inline bool Puttable(const Vec2<i8>& v) {
    if (common::human_count_board[v])
        return false;
    if (common::pet_count_board[v])
        return false;
    for (const auto& d : DIRECTION_VECS) {
        if (common::pet_count_board[v + d])
            return false;
    }
    return true;
}

void MakeAction() {
    // 行動決定
    using common::human_moves;
    using common::next_human_positions;
    using common::starts_left;
    using common::starts_right;

    next_human_positions = common::human_positions;

    rep(i, common::M) { human_moves[i] = '.'; }
    // clang-format off
    constexpr static auto pattern_unit = Board<bool, 10, 10>{
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    };
    // clang-format on
    static auto pattern = Board<bool, 32, 32>();
    if (common::current_turn == 0) {
        rep(i, 32) {
            pattern[{i, 0}] = 1;
            pattern[{i, 31}] = 1;
            pattern[{0, i}] = 1;
            pattern[{31, i}] = 1;
        }
        rep(y, 10) rep(x, 10) rep(y2, 3) rep(x2, 3) pattern[{1 + y + y2 * 10, 1 + x + x2 * 10}] = pattern_unit[{y, x}];
    }

    // ================================================ 0. 行動割り当て ================================================
    static auto setting_assignments = array<i8, 9>{-1, -1, -1, -1, -1, -1, -1, -1, -1};
    struct HumanState {
        enum struct Type {
            NONE,
            MOVING_FOR_SETTING,
            SETTING,
            MOVING,
            WAITING,
        };
        Type type;
        bool setting_left_to_right;
        Vec2<i8> moving_target_position;
        Vec2<i8> setting_target_position;
        i8 assigned_hub;
    };
    static auto human_states = array<HumanState, 10>();
    static auto hub_assignments = array<i8, 9>{-1, -1, -1, -1, -1, -1, -1, -1, -1};

    // clang-format off
    static constexpr auto cell_ids = Board<i8, 32, 32>{
        99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
        99,99, 2, 2, 2, 2,99, 3, 3, 3, 3,99, 4, 4, 4, 4,99, 5, 5, 5, 5,99, 6, 6, 6, 6,99, 7, 7, 7, 7,99,
        99, 1,99, 2, 2, 2,99, 3, 3, 3,99,24,99, 4, 4, 4,99, 5, 5, 5,99,26,99, 6, 6, 6,99, 7, 7, 7,99,99,
        99, 1, 1,99, 2, 2,99, 3, 3,99,24,24,24,99, 4, 4,99, 5, 5,99,26,26,26,99, 6, 6,99, 7, 7,99, 8,99,
        99, 1, 1, 1,99,-1,-1,-1,99,24,24,24,24,24,99,-2,-2,-2,99,26,26,26,26,26,99,-3,-3,-3,99, 8, 8,99,
        99, 1, 1, 1,-1,-1,-1,-1,-1,24,24,24,24,24,-2,-2,-2,-2,-2,26,26,26,26,26,-3,-3,-3,-3,-3, 8, 8,99,
        99,99,99,99,-1,-1,-1,-1,-1,99,99,99,99,99,-2,-2,-2,-2,-2,99,99,99,99,99,-3,-3,-3,-3,-3,99,99,99,
        99, 0, 0, 0,-1,-1,-1,-1,-1,25,25,25,25,25,-2,-2,-2,-2,-2,27,27,27,27,27,-3,-3,-3,-3,-3, 9, 9,99,
        99, 0, 0, 0,99,-1,-1,-1,99,25,25,25,25,25,99,-2,-2,-2,99,27,27,27,27,27,99,-3,-3,-3,99, 9, 9,99,
        99, 0, 0,99,38,38,99,39,39,99,25,25,25,99,40,40,99,41,41,99,27,27,27,99,29,29,99,28,28,99, 9,99,
        99, 0,99,38,38,38,99,39,39,39,99,25,99,40,40,40,99,41,41,41,99,27,99,29,29,29,99,28,28,28,99,99,
        99,99,38,38,38,38,99,39,39,39,39,99,40,40,40,40,99,41,41,41,41,99,29,29,29,29,99,28,28,28,28,99,
        99,23,99,38,38,38,99,39,39,39,99,47,99,40,40,40,99,41,41,41,99,42,99,29,29,29,99,28,28,28,99,99,
        99,23,23,99,38,38,99,39,39,99,47,47,47,99,40,40,99,41,41,99,42,42,42,99,29,29,99,28,28,99,10,99,
        99,23,23,23,99,-4,-4,-4,99,47,47,47,47,47,99,-5,-5,-5,99,42,42,42,42,42,99,-6,-6,-6,99,10,10,99,
        99,23,23,23,-4,-4,-4,-4,-4,47,47,47,47,47,-5,-5,-5,-5,-5,42,42,42,42,42,-6,-6,-6,-6,-6,10,10,99,
        99,99,99,99,-4,-4,-4,-4,-4,99,99,99,99,99,-5,-5,-5,-5,-5,99,99,99,99,99,-6,-6,-6,-6,-6,99,99,99,
        99,22,22,22,-4,-4,-4,-4,-4,46,46,46,46,46,-5,-5,-5,-5,-5,43,43,43,43,43,-6,-6,-6,-6,-6,11,11,99,
        99,22,22,22,99,-4,-4,-4,99,46,46,46,46,46,99,-5,-5,-5,99,43,43,43,43,43,99,-6,-6,-6,99,11,11,99,
        99,22,22,99,36,36,99,37,37,99,46,46,46,99,45,45,99,44,44,99,43,43,43,99,31,31,99,30,30,99,11,99,
        99,22,99,36,36,36,99,37,37,37,99,46,99,45,45,45,99,44,44,44,99,43,99,31,31,31,99,30,30,30,99,99,
        99,99,36,36,36,36,99,37,37,37,37,99,45,45,45,45,99,44,44,44,44,99,31,31,31,31,99,30,30,30,30,99,
        99,21,99,36,36,36,99,37,37,37,99,35,99,45,45,45,99,44,44,44,99,33,99,31,31,31,99,30,30,30,99,99,
        99,21,21,99,36,36,99,37,37,99,35,35,35,99,45,45,99,44,44,99,33,33,33,99,31,31,99,30,30,99,12,99,
        99,21,21,21,99,-7,-7,-7,99,35,35,35,35,35,99,-8,-8,-8,99,33,33,33,33,33,99,-9,-9,-9,99,12,12,99,
        99,21,21,21,-7,-7,-7,-7,-7,35,35,35,35,35,-8,-8,-8,-8,-8,33,33,33,33,33,-9,-9,-9,-9,-9,12,12,99,
        99,99,99,99,-7,-7,-7,-7,-7,99,99,99,99,99,-8,-8,-8,-8,-8,99,99,99,99,99,-9,-9,-9,-9,-9,99,99,99,
        99,20,20,20,-7,-7,-7,-7,-7,34,34,34,34,34,-8,-8,-8,-8,-8,32,32,32,32,32,-9,-9,-9,-9,-9,13,13,99,
        99,20,20,20,99,-7,-7,-7,99,34,34,34,34,34,99,-8,-8,-8,99,32,32,32,32,32,99,-9,-9,-9,99,13,13,99,
        99,20,20,99,19,19,99,18,18,99,34,34,34,99,17,17,99,16,16,99,32,32,32,99,15,15,99,14,14,99,13,99,
        99,20,99,19,19,19,99,18,18,18,99,34,99,17,17,17,99,16,16,16,99,32,99,15,15,15,99,14,14,14,99,99,
        99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
    };
    static constexpr auto cell_id_to_hub_ids = array<array<i8, 2>, 48>{ array<i8, 2>
        {0}, {0}, {0}, {0}, // 0-
        {1}, {1},
        {2}, {2}, {2}, {2},
        {5}, {5},
        {8}, {8}, {8}, {8},
        {7}, {7},
        {6}, {6}, {6}, {6},
        {3}, {3},
        {0, 1}, {0, 1}, // 24-
        {1, 2}, {1, 2},
        {2, 5}, {2, 5},
        {5, 8}, {5, 8},
        {8, 7}, {8, 7},
        {7, 6}, {7, 6},
        {6, 3}, {6, 3},
        {3, 0}, {3, 0},
        {4, 1}, {4, 1}, // 40-
        {4, 5}, {4, 5},
        {4, 7}, {4, 7},
        {4, 3}, {4, 3},
    };
    // clang-format on
    static constexpr auto hub_centers = array<Vec2<i8>, 9>{
        Vec2<i8>{6, 6}, {6, 16}, {6, 26}, {16, 6}, {16, 16}, {16, 26}, {26, 6}, {26, 16}, {26, 26},
    };
    rep(idx_human, common::M) {
        if (human_states[idx_human].type == HumanState::Type::MOVING) {
            human_states[idx_human].type = HumanState::Type::NONE;
            hub_assignments[human_states[idx_human].assigned_hub] = -1;
            human_states[idx_human].assigned_hub = -1;
        } else if (human_states[idx_human].type == HumanState::Type::WAITING) {
            rep(i, common::N) {
                if (features::distances_from_each_human[idx_human][common::pet_positions[i]] <= 12) {
                    goto ok;
                }
            }
            human_states[idx_human].type = HumanState::Type::NONE;
            hub_assignments[human_states[idx_human].assigned_hub] = -1;
            human_states[idx_human].assigned_hub = -1;
        ok:;
        }
        if (human_states[idx_human].type == HumanState::Type::NONE) {
            auto best_assignment = -1;
            auto best_assignment_direction_is_right = false;
            auto best_distance = 999;
            rep(i, 9) {
                if (setting_assignments[i] == -1) {
                    if (chmin(best_distance, features::distances_from_each_human[idx_human][starts_left[i]])) {
                        best_assignment = i;
                        best_assignment_direction_is_right = true;
                    }
                    if (chmin(best_distance, features::distances_from_each_human[idx_human][starts_right[i]])) {
                        best_assignment = i;
                        best_assignment_direction_is_right = false;
                    }
                }
            }
            if (best_assignment != -1) {
                setting_assignments[best_assignment] = idx_human;
                human_states[idx_human].type = HumanState::Type::MOVING_FOR_SETTING;
                human_states[idx_human].setting_left_to_right = best_assignment_direction_is_right;
                human_states[idx_human].moving_target_position = (best_assignment_direction_is_right ? starts_left : starts_right)[best_assignment];
                human_states[idx_human].setting_target_position = (best_assignment_direction_is_right ? starts_right : starts_left)[best_assignment];

                if (best_distance == 0) {
                    human_states[idx_human].type = HumanState::Type::SETTING;
                }
                continue;
            }
            // 今居る位置に意味がないなら自由になる
            // 自由なら、最も近い、意味のある、ほかに人の居ない位置に移動
            // 自由でないなら、閉じ込め作業
            // 閉じ込めは、両側とも準備ができている中で近いものを最優先、次に準備ができている遠いもの
            auto best_hub = -1;
            auto best_hub_distance = 999;
            rep(i, common::N) {
                const auto& pet_pos = common::pet_positions[i];
                if (features::distances_from_each_human[idx_human][pet_pos] == 999)
                    continue;
                const auto& pet_cell = cell_ids[pet_pos];
                if (pet_cell < 0) {
                    const auto hub = ~pet_cell;
                    if (hub_assignments[hub] == -1) {
                        if (chmin(best_hub_distance, features::distances_from_each_human[idx_human][hub_centers[hub]]))
                            best_hub = hub;
                    }
                } else if (pet_cell < 24) {
                    const auto& hub = cell_id_to_hub_ids[pet_cell][0];
                    if (hub_assignments[hub] == -1) {
                        if (chmin(best_hub_distance, features::distances_from_each_human[idx_human][hub_centers[hub]]))
                            best_hub = hub;
                    }
                } else {
                    rep(idx_hubs, 2) {
                        const auto& hub = cell_id_to_hub_ids[pet_cell][idx_hubs];
                        if (hub_assignments[hub] == -1) {
                            if (chmin(best_hub_distance, features::distances_from_each_human[idx_human][hub_centers[hub]]))
                                best_hub = hub;
                        }
                    }
                }
            }
            if (best_hub != -1) {
                hub_assignments[best_hub] = idx_human;
                human_states[idx_human].moving_target_position = hub_centers[best_hub];
                human_states[idx_human].assigned_hub = best_hub;
                if (best_hub_distance <= 2) {
                    human_states[idx_human].type = HumanState::Type::WAITING;
                } else {
                    human_states[idx_human].type = HumanState::Type::MOVING;
                }
            }
        }
    }

    // ここに WAITING 同士の通信？

    rep(idx_human, 9) {
        if (human_states[idx_human].type == HumanState::Type::MOVING_FOR_SETTING) {
            // ================================================ 1. 移動 ================================================
            static auto distance_board = Board<short, 32, 32>();
            const auto& target_position = human_states[idx_human].moving_target_position;
            bfs(common::fence_board, target_position, distance_board);
            rep(i, 4) {
                const auto& v = common::human_positions[idx_human];
                const auto u = v + DIRECTION_VECS[i];
                if (distance_board[u] < distance_board[v]) {
                    human_moves[idx_human] = "UDLR"[i];
                    next_human_positions[idx_human] = u;
                }
            }
            if (next_human_positions[idx_human] == target_position) {
                human_states[idx_human].type = HumanState::Type::SETTING;
            }
        } else if (human_states[idx_human].type == HumanState::Type::SETTING) {
            // ================================================ 2. 設置 ================================================
            const auto& v = common::human_positions[idx_human];
            const auto& l_to_r = human_states[idx_human].setting_left_to_right;
            auto n_remaining_put_place = 0;
            for (const auto& i : l_to_r ? array<int, 3>{0, 1, 2} : array<int, 3>{0, 1, 3}) {
                const auto u = v + DIRECTION_VECS[i];
                if (pattern[u] && !common::fence_board[u]) {
                    n_remaining_put_place++;
                    if (Puttable(u)) {
                        human_moves[idx_human] = "udlr"[i];
                    }
                }
            }
            if (n_remaining_put_place == 0) {
                // 移動
                human_moves[idx_human] = l_to_r ? 'R' : 'L';
                next_human_positions[idx_human] = common::human_positions[idx_human] + (l_to_r ? Vec2<i8>{0, 1} : Vec2<i8>{0, -1});
            } else if (n_remaining_put_place == 1 && human_moves[idx_human] != '.') {
                // 終了判定
                if (common::human_positions[idx_human] == human_states[idx_human].setting_target_position) {
                    human_states[idx_human].type = HumanState::Type::NONE;
                }
            }
        } else if (human_states[idx_human].type == HumanState::Type::MOVING) {
            // ================================================ 3. 移動 ================================================
            static auto distance_board = Board<short, 32, 32>();
            const auto& target_position = human_states[idx_human].moving_target_position;
            bfs(common::fence_board, target_position, distance_board);
            rep(i, 4) {
                const auto& v = common::human_positions[idx_human];
                const auto u = v + DIRECTION_VECS[i];
                if (distance_board[u] < distance_board[v]) {
                    human_moves[idx_human] = "UDLR"[i];
                    next_human_positions[idx_human] = u;
                }
            }
            if (distance_board[next_human_positions[idx_human]] == 2) {
                human_states[idx_human].type = HumanState::Type::WAITING;
            }
        }
    }
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
        MakeAction();
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
