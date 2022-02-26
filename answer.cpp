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

#include <atcoder/dsu>

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

// 回帰木
template <int max_size = 32> struct DecisionTree {
    int node_count;                                     // 内部ノード + 葉  // これ別に無くてもいい？
    array<int, max_size> children_left, children_right; // node_count 個の要素
    array<int, max_size> feature;                       // node_count 個の要素
    array<double, max_size> threshold;                  // node_count 個の要素
    array<double, max_size> value;                      // node_count 個の要素
    template <typename Array> double Predict(const Array& x) const {
        auto node = 0;
        while (children_left[node] != -1) {
            node = (x[feature[node]] <= threshold[node] ? children_left : children_right)[node];
        }
        return value[node];
    }
};

// ランダムフォレスト回帰
template <int n_trees, int max_tree_size> struct RandomForest {
    array<DecisionTree<max_tree_size>, n_trees> trees;
    template <typename Array> double Predict(const Array& x) const {
        auto res = 0.0;
        for (const auto& tree : trees) {
            res += tree.Predict(x);
        }
        return res * (1.0 / (double)n_trees);
    }
};

// ===========================================================================================================

enum struct Directions { U, D, L, R, NONE };
constexpr auto DIRECTION_VECS = array<Vec2<int>, 4>{Vec2<int>{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
enum struct PetType { NONE, COW, PIG, RABBIT, DOG, CAT };
using i8 = signed char;

auto SHORT_BAR = false;

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
            if (fence_board[v]) {
                cout << "#WTF" << endl;
            }
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

inline bool TightPuttable(const Vec2<i8>& v) {
    if (common::human_count_board[v])
        return false;
    if (common::pet_count_board[v])
        return false;
    for (const auto& d : DIRECTION_VECS) {
        if (common::pet_count_board[v + d])
            return false;
    }
    rep(i, common::M) {
        if (common::next_human_positions[i] == v)
            return false;
    }
    return true;
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
    auto human_cnt = 0;
    for (const auto& d : DIRECTION_VECS) {
        human_cnt += common::human_count_board[v + d];
    }
    if (human_cnt >= 2)
        return false; // 安全策…
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
    static auto pattern_unit = SHORT_BAR ? 
        Board<bool, 10, 10>{
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
        } : Board<bool, 10, 10>{
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
            CATCHING,
        };
        Type type;
        bool setting_left_to_right;
        Vec2<i8> moving_target_position;
        Vec2<i8> setting_target_position;
        i8 assigned_hub;
        i8 assigned_pet;
    };
    static auto human_states = array<HumanState, 10>();
    if (common::current_turn == 0) {
        rep(i, 10) human_states[i].assigned_hub = -1;
    }
    static auto hub_assignments = array<i8, 9>{-1, -1, -1, -1, -1, -1, -1, -1, -1};

    // clang-format off
    static auto cell_ids = SHORT_BAR ? Board<i8, 32, 32>{
        99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
        99,99, 2, 2, 2, 2,99, 3, 3, 3, 3,99, 4, 4, 4, 4,99, 5, 5, 5, 5,99, 6, 6, 6, 6,99, 7, 7, 7, 7,99,
        99, 1,99, 2, 2, 2,99, 3, 3, 3,99,24,99, 4, 4, 4,99, 5, 5, 5,99,26,99, 6, 6, 6,99, 7, 7, 7,99,99,
        99, 1, 1,99, 2, 2,-1, 3, 3,99,24,24,24,99, 4, 4,-2, 5, 5,99,26,26,26,99, 6, 6,-3, 7, 7,99, 8,99,
        99, 1, 1, 1,99,-1,-1,-1,99,24,24,24,24,24,99,-2,-2,-2,99,26,26,26,26,26,99,-3,-3,-3,99, 8, 8,99,
        99, 1, 1, 1,-1,-1,-1,-1,-1,24,24,24,24,24,-2,-2,-2,-2,-2,26,26,26,26,26,-3,-3,-3,-3,-3, 8, 8,99,
        99,99,99,-1,-1,-1,-1,-1,-1,-1,99,99,99,-2,-2,-2,-2,-2,-2,-2,99,99,99,-3,-3,-3,-3,-3,-3,-3,99,99,
        99, 0, 0, 0,-1,-1,-1,-1,-1,25,25,25,25,25,-2,-2,-2,-2,-2,27,27,27,27,27,-3,-3,-3,-3,-3, 9, 9,99,
        99, 0, 0, 0,99,-1,-1,-1,99,25,25,25,25,25,99,-2,-2,-2,99,27,27,27,27,27,99,-3,-3,-3,99, 9, 9,99,
        99, 0, 0,99,38,38,-1,39,39,99,25,25,25,99,40,40,-2,41,41,99,27,27,27,99,29,29,-3,28,28,99, 9,99,
        99, 0,99,38,38,38,99,39,39,39,99,25,99,40,40,40,99,41,41,41,99,27,99,29,29,29,99,28,28,28,99,99,
        99,99,38,38,38,38,99,39,39,39,39,99,40,40,40,40,99,41,41,41,41,99,29,29,29,29,99,28,28,28,28,99,
        99,23,99,38,38,38,99,39,39,39,99,47,99,40,40,40,99,41,41,41,99,42,99,29,29,29,99,28,28,28,99,99,
        99,23,23,99,38,38,-4,39,39,99,47,47,47,99,40,40,-5,41,41,99,42,42,42,99,29,29,-6,28,28,99,10,99,
        99,23,23,23,99,-4,-4,-4,99,47,47,47,47,47,99,-5,-5,-5,99,42,42,42,42,42,99,-6,-6,-6,99,10,10,99,
        99,23,23,23,-4,-4,-4,-4,-4,47,47,47,47,47,-5,-5,-5,-5,-5,42,42,42,42,42,-6,-6,-6,-6,-6,10,10,99,
        99,99,99,-4,-4,-4,-4,-4,-4,-4,99,99,99,-5,-5,-5,-5,-5,-5,-5,99,99,99,-6,-6,-6,-6,-6,-6,-6,99,99,
        99,22,22,22,-4,-4,-4,-4,-4,46,46,46,46,46,-5,-5,-5,-5,-5,43,43,43,43,43,-6,-6,-6,-6,-6,11,11,99,
        99,22,22,22,99,-4,-4,-4,99,46,46,46,46,46,99,-5,-5,-5,99,43,43,43,43,43,99,-6,-6,-6,99,11,11,99,
        99,22,22,99,36,36,-4,37,37,99,46,46,46,99,45,45,-5,44,44,99,43,43,43,99,31,31,-6,30,30,99,11,99,
        99,22,99,36,36,36,99,37,37,37,99,46,99,45,45,45,99,44,44,44,99,43,99,31,31,31,99,30,30,30,99,99,
        99,99,36,36,36,36,99,37,37,37,37,99,45,45,45,45,99,44,44,44,44,99,31,31,31,31,99,30,30,30,30,99,
        99,21,99,36,36,36,99,37,37,37,99,35,99,45,45,45,99,44,44,44,99,33,99,31,31,31,99,30,30,30,99,99,
        99,21,21,99,36,36,-7,37,37,99,35,35,35,99,45,45,-8,44,44,99,33,33,33,99,31,31,-9,30,30,99,12,99,
        99,21,21,21,99,-7,-7,-7,99,35,35,35,35,35,99,-8,-8,-8,99,33,33,33,33,33,99,-9,-9,-9,99,12,12,99,
        99,21,21,21,-7,-7,-7,-7,-7,35,35,35,35,35,-8,-8,-8,-8,-8,33,33,33,33,33,-9,-9,-9,-9,-9,12,12,99,
        99,99,99,-7,-7,-7,-7,-7,-7,-7,99,99,99,-8,-8,-8,-8,-8,-8,-8,99,99,99,-9,-9,-9,-9,-9,-9,-9,99,99,
        99,20,20,20,-7,-7,-7,-7,-7,34,34,34,34,34,-8,-8,-8,-8,-8,32,32,32,32,32,-9,-9,-9,-9,-9,13,13,99,
        99,20,20,20,99,-7,-7,-7,99,34,34,34,34,34,99,-8,-8,-8,99,32,32,32,32,32,99,-9,-9,-9,99,13,13,99,
        99,20,20,99,19,19,-7,18,18,99,34,34,34,99,17,17,-8,16,16,99,32,32,32,99,15,15,-9,14,14,99,13,99,
        99,20,99,19,19,19,99,18,18,18,99,34,99,17,17,17,99,16,16,16,99,32,99,15,15,15,99,14,14,14,99,99,
        99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
    } : Board<i8, 32, 32>{
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
    static auto cell_neighboring_fences = SHORT_BAR ? array<array<Vec2<i8>, 16>, 48>{array<Vec2<i8>, 16>
        {Vec2<i8>{ 6, 1},{ 6, 2},{ 7, 0},{ 8, 0},{ 8, 4},{ 9, 3},{ 9, 0},{ 9, 3},{10, 2},{10, 0},{10, 2},{11, 1},},
        {Vec2<i8>{ 1, 1},{ 2, 0},{ 2, 2},{ 2, 2},{ 3, 0},{ 3, 3},{ 3, 3},{ 4, 0},{ 4, 4},{ 5, 0},{ 6, 1},{ 6, 2},},
        {Vec2<i8>{ 0, 2},{ 0, 3},{ 0, 4},{ 0, 5},{ 1, 1},{ 2, 2},{ 1, 6},{ 2, 2},{ 3, 3},{ 2, 6},{ 3, 3},{ 4, 4},},
        {Vec2<i8>{ 0, 7},{ 0, 8},{ 0, 9},{ 0,10},{ 1, 6},{ 1,11},{ 2,10},{ 2, 6},{ 2,10},{ 3, 9},{ 3, 9},{ 4, 8},},
        {Vec2<i8>{ 0,12},{ 0,13},{ 0,14},{ 0,15},{ 1,11},{ 2,12},{ 1,16},{ 2,12},{ 3,13},{ 2,16},{ 3,13},{ 4,14},},
        {Vec2<i8>{ 0,17},{ 0,18},{ 0,19},{ 0,20},{ 1,16},{ 1,21},{ 2,20},{ 2,16},{ 2,20},{ 3,19},{ 3,19},{ 4,18},},
        {Vec2<i8>{ 0,22},{ 0,23},{ 0,24},{ 0,25},{ 1,21},{ 2,22},{ 1,26},{ 2,22},{ 3,23},{ 2,26},{ 3,23},{ 4,24},},
        {Vec2<i8>{ 0,27},{ 0,28},{ 0,29},{ 0,30},{ 1,26},{ 1,31},{ 2,30},{ 2,26},{ 2,30},{ 3,29},{ 3,29},{ 4,28},},
        {Vec2<i8>{ 2,30},{ 3,29},{ 3,29},{ 3,31},{ 4,28},{ 4,31},{ 5,31},{ 6,30},},
        {Vec2<i8>{ 6,30},{ 7,31},{ 8,28},{ 9,29},{ 8,31},{ 9,29},{ 9,31},{10,30},},
        {Vec2<i8>{12,30},{13,29},{13,29},{13,31},{14,28},{14,31},{15,31},{16,30},},
        {Vec2<i8>{16,30},{17,31},{18,28},{19,29},{18,31},{19,29},{19,31},{20,30},},
        {Vec2<i8>{22,30},{23,29},{23,29},{23,31},{24,28},{24,31},{25,31},{26,30},},
        {Vec2<i8>{26,30},{27,31},{28,28},{29,29},{28,31},{29,29},{29,31},{30,30},},
        {Vec2<i8>{28,28},{29,29},{29,29},{30,26},{31,27},{31,28},{30,30},{31,29},},
        {Vec2<i8>{28,24},{29,23},{29,23},{30,22},{31,23},{31,24},{30,26},{31,25},},
        {Vec2<i8>{28,18},{29,19},{29,19},{30,16},{31,17},{31,18},{30,20},{31,19},},
        {Vec2<i8>{28,14},{29,13},{29,13},{30,12},{31,13},{31,14},{30,16},{31,15},},
        {Vec2<i8>{28, 8},{29, 9},{29, 9},{30, 6},{31, 7},{31, 8},{30,10},{31, 9},},
        {Vec2<i8>{28, 4},{29, 3},{29, 3},{30, 2},{31, 3},{31, 4},{30, 6},{31, 5},},
        {Vec2<i8>{26, 1},{26, 2},{27, 0},{28, 0},{28, 4},{29, 3},{29, 0},{29, 3},{30, 2},{30, 0},{30, 2},{31, 1},},
        {Vec2<i8>{21, 1},{22, 0},{22, 2},{22, 2},{23, 0},{23, 3},{23, 3},{24, 0},{24, 4},{25, 0},{26, 1},{26, 2},},
        {Vec2<i8>{16, 1},{16, 2},{17, 0},{18, 0},{18, 4},{19, 3},{19, 0},{19, 3},{20, 2},{20, 0},{20, 2},{21, 1},},
        {Vec2<i8>{11, 1},{12, 0},{12, 2},{12, 2},{13, 0},{13, 3},{13, 3},{14, 0},{14, 4},{15, 0},{16, 1},{16, 2},},
        {Vec2<i8>{ 1,11},{ 2,10},{ 2,10},{ 2,12},{ 2,12},{ 3, 9},{ 3, 9},{ 3,13},{ 3,13},{ 4, 8},{ 4,14},{ 6,10},{ 6,11},{ 6,12},},
        {Vec2<i8>{ 6,10},{ 6,11},{ 6,12},{ 8, 8},{ 9, 9},{ 8,14},{ 9,13},{ 9, 9},{10,10},{ 9,13},{10,12},{10,10},{10,12},{11,11},},
        {Vec2<i8>{ 1,21},{ 2,20},{ 2,20},{ 2,22},{ 2,22},{ 3,19},{ 3,19},{ 3,23},{ 3,23},{ 4,18},{ 4,24},{ 6,20},{ 6,21},{ 6,22},},
        {Vec2<i8>{ 6,20},{ 6,21},{ 6,22},{ 8,18},{ 9,19},{ 8,24},{ 9,23},{ 9,19},{10,20},{ 9,23},{10,22},{10,20},{10,22},{11,21},},
        {Vec2<i8>{ 8,28},{ 9,29},{ 9,29},{10,26},{10,30},{10,30},{11,26},{11,31},{12,30},{12,26},{12,30},{13,29},{13,29},{14,28},},
        {Vec2<i8>{ 8,24},{ 9,23},{ 9,23},{10,22},{10,22},{10,26},{11,21},{12,22},{11,26},{12,22},{13,23},{12,26},{13,23},{14,24},},
        {Vec2<i8>{18,28},{19,29},{19,29},{20,26},{20,30},{20,30},{21,26},{21,31},{22,30},{22,26},{22,30},{23,29},{23,29},{24,28},},
        {Vec2<i8>{18,24},{19,23},{19,23},{20,22},{20,22},{20,26},{21,21},{22,22},{21,26},{22,22},{23,23},{22,26},{23,23},{24,24},},
        {Vec2<i8>{26,20},{26,21},{26,22},{28,18},{29,19},{28,24},{29,23},{29,19},{30,20},{29,23},{30,22},{30,20},{30,22},{31,21},},
        {Vec2<i8>{21,21},{22,20},{22,20},{22,22},{22,22},{23,19},{23,19},{23,23},{23,23},{24,18},{24,24},{26,20},{26,21},{26,22},},
        {Vec2<i8>{26,10},{26,11},{26,12},{28, 8},{29, 9},{28,14},{29,13},{29, 9},{30,10},{29,13},{30,12},{30,10},{30,12},{31,11},},
        {Vec2<i8>{21,11},{22,10},{22,10},{22,12},{22,12},{23, 9},{23, 9},{23,13},{23,13},{24, 8},{24,14},{26,10},{26,11},{26,12},},
        {Vec2<i8>{18, 4},{19, 3},{19, 3},{20, 2},{20, 2},{20, 6},{21, 1},{22, 2},{21, 6},{22, 2},{23, 3},{22, 6},{23, 3},{24, 4},},
        {Vec2<i8>{18, 8},{19, 9},{19, 9},{20, 6},{20,10},{20,10},{21, 6},{21,11},{22,10},{22, 6},{22,10},{23, 9},{23, 9},{24, 8},},
        {Vec2<i8>{ 8, 4},{ 9, 3},{ 9, 3},{10, 2},{10, 2},{10, 6},{11, 1},{12, 2},{11, 6},{12, 2},{13, 3},{12, 6},{13, 3},{14, 4},},
        {Vec2<i8>{ 8, 8},{ 9, 9},{ 9, 9},{10, 6},{10,10},{10,10},{11, 6},{11,11},{12,10},{12, 6},{12,10},{13, 9},{13, 9},{14, 8},},
        {Vec2<i8>{ 8,14},{ 9,13},{ 9,13},{10,12},{10,12},{10,16},{11,11},{12,12},{11,16},{12,12},{13,13},{12,16},{13,13},{14,14},},
        {Vec2<i8>{ 8,18},{ 9,19},{ 9,19},{10,16},{10,20},{10,20},{11,16},{11,21},{12,20},{12,16},{12,20},{13,19},{13,19},{14,18},},
        {Vec2<i8>{11,21},{12,20},{12,20},{12,22},{12,22},{13,19},{13,19},{13,23},{13,23},{14,18},{14,24},{16,20},{16,21},{16,22},},
        {Vec2<i8>{16,20},{16,21},{16,22},{18,18},{19,19},{18,24},{19,23},{19,19},{20,20},{19,23},{20,22},{20,20},{20,22},{21,21},},
        {Vec2<i8>{18,18},{19,19},{19,19},{20,16},{20,20},{20,20},{21,16},{21,21},{22,20},{22,16},{22,20},{23,19},{23,19},{24,18},},
        {Vec2<i8>{18,14},{19,13},{19,13},{20,12},{20,12},{20,16},{21,11},{22,12},{21,16},{22,12},{23,13},{22,16},{23,13},{24,14},},
        {Vec2<i8>{16,10},{16,11},{16,12},{18, 8},{19, 9},{18,14},{19,13},{19, 9},{20,10},{19,13},{20,12},{20,10},{20,12},{21,11},},
        {Vec2<i8>{11,11},{12,10},{12,10},{12,12},{12,12},{13, 9},{13, 9},{13,13},{13,13},{14, 8},{14,14},{16,10},{16,11},{16,12},},
    } : array<array<Vec2<i8>, 16>, 48>{array<Vec2<i8>, 16>
        {Vec2<i8>{ 6, 1},{ 6, 2},{ 6, 3},{ 7, 0},{ 8, 0},{ 8, 4},{ 9, 3},{ 9, 0},{ 9, 3},{10, 2},{10, 0},{10, 2},{11, 1},},
        {Vec2<i8>{ 1, 1},{ 2, 0},{ 2, 2},{ 2, 2},{ 3, 0},{ 3, 3},{ 3, 3},{ 4, 0},{ 4, 4},{ 5, 0},{ 6, 1},{ 6, 2},{ 6, 3},},
        {Vec2<i8>{ 0, 2},{ 0, 3},{ 0, 4},{ 0, 5},{ 1, 1},{ 2, 2},{ 1, 6},{ 2, 2},{ 3, 3},{ 2, 6},{ 3, 3},{ 4, 4},{ 3, 6},},
        {Vec2<i8>{ 0, 7},{ 0, 8},{ 0, 9},{ 0,10},{ 1, 6},{ 1,11},{ 2,10},{ 2, 6},{ 2,10},{ 3, 9},{ 3, 6},{ 3, 9},{ 4, 8},},
        {Vec2<i8>{ 0,12},{ 0,13},{ 0,14},{ 0,15},{ 1,11},{ 2,12},{ 1,16},{ 2,12},{ 3,13},{ 2,16},{ 3,13},{ 4,14},{ 3,16},},
        {Vec2<i8>{ 0,17},{ 0,18},{ 0,19},{ 0,20},{ 1,16},{ 1,21},{ 2,20},{ 2,16},{ 2,20},{ 3,19},{ 3,16},{ 3,19},{ 4,18},},
        {Vec2<i8>{ 0,22},{ 0,23},{ 0,24},{ 0,25},{ 1,21},{ 2,22},{ 1,26},{ 2,22},{ 3,23},{ 2,26},{ 3,23},{ 4,24},{ 3,26},},
        {Vec2<i8>{ 0,27},{ 0,28},{ 0,29},{ 0,30},{ 1,26},{ 1,31},{ 2,30},{ 2,26},{ 2,30},{ 3,29},{ 3,26},{ 3,29},{ 4,28},},
        {Vec2<i8>{ 2,30},{ 3,29},{ 3,29},{ 3,31},{ 4,28},{ 4,31},{ 6,29},{ 5,31},{ 6,30},},
        {Vec2<i8>{ 6,29},{ 6,30},{ 7,31},{ 8,28},{ 9,29},{ 8,31},{ 9,29},{ 9,31},{10,30},},
        {Vec2<i8>{12,30},{13,29},{13,29},{13,31},{14,28},{14,31},{16,29},{15,31},{16,30},},
        {Vec2<i8>{16,29},{16,30},{17,31},{18,28},{19,29},{18,31},{19,29},{19,31},{20,30},},
        {Vec2<i8>{22,30},{23,29},{23,29},{23,31},{24,28},{24,31},{26,29},{25,31},{26,30},},
        {Vec2<i8>{26,29},{26,30},{27,31},{28,28},{29,29},{28,31},{29,29},{29,31},{30,30},},
        {Vec2<i8>{28,28},{29,26},{29,29},{29,29},{30,26},{31,27},{31,28},{30,30},{31,29},},
        {Vec2<i8>{28,24},{29,23},{29,23},{29,26},{30,22},{31,23},{31,24},{30,26},{31,25},},
        {Vec2<i8>{28,18},{29,16},{29,19},{29,19},{30,16},{31,17},{31,18},{30,20},{31,19},},
        {Vec2<i8>{28,14},{29,13},{29,13},{29,16},{30,12},{31,13},{31,14},{30,16},{31,15},},
        {Vec2<i8>{28, 8},{29, 6},{29, 9},{29, 9},{30, 6},{31, 7},{31, 8},{30,10},{31, 9},},
        {Vec2<i8>{28, 4},{29, 3},{29, 3},{29, 6},{30, 2},{31, 3},{31, 4},{30, 6},{31, 5},},
        {Vec2<i8>{26, 1},{26, 2},{26, 3},{27, 0},{28, 0},{28, 4},{29, 3},{29, 0},{29, 3},{30, 2},{30, 0},{30, 2},{31, 1},},
        {Vec2<i8>{21, 1},{22, 0},{22, 2},{22, 2},{23, 0},{23, 3},{23, 3},{24, 0},{24, 4},{25, 0},{26, 1},{26, 2},{26, 3},},
        {Vec2<i8>{16, 1},{16, 2},{16, 3},{17, 0},{18, 0},{18, 4},{19, 3},{19, 0},{19, 3},{20, 2},{20, 0},{20, 2},{21, 1},},
        {Vec2<i8>{11, 1},{12, 0},{12, 2},{12, 2},{13, 0},{13, 3},{13, 3},{14, 0},{14, 4},{15, 0},{16, 1},{16, 2},{16, 3},},
        {Vec2<i8>{ 1,11},{ 2,10},{ 2,10},{ 2,12},{ 2,12},{ 3, 9},{ 3, 9},{ 3,13},{ 3,13},{ 4, 8},{ 4,14},{ 6, 9},{ 6,10},{ 6,11},{ 6,12},{ 6,13},},
        {Vec2<i8>{ 6, 9},{ 6,10},{ 6,11},{ 6,12},{ 6,13},{ 8, 8},{ 9, 9},{ 8,14},{ 9,13},{ 9, 9},{10,10},{ 9,13},{10,12},{10,10},{10,12},{11,11},},
        {Vec2<i8>{ 1,21},{ 2,20},{ 2,20},{ 2,22},{ 2,22},{ 3,19},{ 3,19},{ 3,23},{ 3,23},{ 4,18},{ 4,24},{ 6,19},{ 6,20},{ 6,21},{ 6,22},{ 6,23},},
        {Vec2<i8>{ 6,19},{ 6,20},{ 6,21},{ 6,22},{ 6,23},{ 8,18},{ 9,19},{ 8,24},{ 9,23},{ 9,19},{10,20},{ 9,23},{10,22},{10,20},{10,22},{11,21},},
        {Vec2<i8>{ 8,28},{ 9,26},{ 9,29},{ 9,29},{10,26},{10,30},{10,30},{11,26},{11,31},{12,30},{12,26},{12,30},{13,29},{13,26},{13,29},{14,28},},
        {Vec2<i8>{ 8,24},{ 9,23},{ 9,23},{ 9,26},{10,22},{10,22},{10,26},{11,21},{12,22},{11,26},{12,22},{13,23},{12,26},{13,23},{14,24},{13,26},},
        {Vec2<i8>{18,28},{19,26},{19,29},{19,29},{20,26},{20,30},{20,30},{21,26},{21,31},{22,30},{22,26},{22,30},{23,29},{23,26},{23,29},{24,28},},
        {Vec2<i8>{18,24},{19,23},{19,23},{19,26},{20,22},{20,22},{20,26},{21,21},{22,22},{21,26},{22,22},{23,23},{22,26},{23,23},{24,24},{23,26},},
        {Vec2<i8>{26,19},{26,20},{26,21},{26,22},{26,23},{28,18},{29,19},{28,24},{29,23},{29,19},{30,20},{29,23},{30,22},{30,20},{30,22},{31,21},},
        {Vec2<i8>{21,21},{22,20},{22,20},{22,22},{22,22},{23,19},{23,19},{23,23},{23,23},{24,18},{24,24},{26,19},{26,20},{26,21},{26,22},{26,23},},
        {Vec2<i8>{26, 9},{26,10},{26,11},{26,12},{26,13},{28, 8},{29, 9},{28,14},{29,13},{29, 9},{30,10},{29,13},{30,12},{30,10},{30,12},{31,11},},
        {Vec2<i8>{21,11},{22,10},{22,10},{22,12},{22,12},{23, 9},{23, 9},{23,13},{23,13},{24, 8},{24,14},{26, 9},{26,10},{26,11},{26,12},{26,13},},
        {Vec2<i8>{18, 4},{19, 3},{19, 3},{19, 6},{20, 2},{20, 2},{20, 6},{21, 1},{22, 2},{21, 6},{22, 2},{23, 3},{22, 6},{23, 3},{24, 4},{23, 6},},
        {Vec2<i8>{18, 8},{19, 6},{19, 9},{19, 9},{20, 6},{20,10},{20,10},{21, 6},{21,11},{22,10},{22, 6},{22,10},{23, 9},{23, 6},{23, 9},{24, 8},},
        {Vec2<i8>{ 8, 4},{ 9, 3},{ 9, 3},{ 9, 6},{10, 2},{10, 2},{10, 6},{11, 1},{12, 2},{11, 6},{12, 2},{13, 3},{12, 6},{13, 3},{14, 4},{13, 6},},
        {Vec2<i8>{ 8, 8},{ 9, 6},{ 9, 9},{ 9, 9},{10, 6},{10,10},{10,10},{11, 6},{11,11},{12,10},{12, 6},{12,10},{13, 9},{13, 6},{13, 9},{14, 8},},
        {Vec2<i8>{ 8,14},{ 9,13},{ 9,13},{ 9,16},{10,12},{10,12},{10,16},{11,11},{12,12},{11,16},{12,12},{13,13},{12,16},{13,13},{14,14},{13,16},},
        {Vec2<i8>{ 8,18},{ 9,16},{ 9,19},{ 9,19},{10,16},{10,20},{10,20},{11,16},{11,21},{12,20},{12,16},{12,20},{13,19},{13,16},{13,19},{14,18},},
        {Vec2<i8>{11,21},{12,20},{12,20},{12,22},{12,22},{13,19},{13,19},{13,23},{13,23},{14,18},{14,24},{16,19},{16,20},{16,21},{16,22},{16,23},},
        {Vec2<i8>{16,19},{16,20},{16,21},{16,22},{16,23},{18,18},{19,19},{18,24},{19,23},{19,19},{20,20},{19,23},{20,22},{20,20},{20,22},{21,21},},
        {Vec2<i8>{18,18},{19,16},{19,19},{19,19},{20,16},{20,20},{20,20},{21,16},{21,21},{22,20},{22,16},{22,20},{23,19},{23,16},{23,19},{24,18},},
        {Vec2<i8>{18,14},{19,13},{19,13},{19,16},{20,12},{20,12},{20,16},{21,11},{22,12},{21,16},{22,12},{23,13},{22,16},{23,13},{24,14},{23,16},},
        {Vec2<i8>{16, 9},{16,10},{16,11},{16,12},{16,13},{18, 8},{19, 9},{18,14},{19,13},{19, 9},{20,10},{19,13},{20,12},{20,10},{20,12},{21,11},},
        {Vec2<i8>{11,11},{12,10},{12,10},{12,12},{12,12},{13, 9},{13, 9},{13,13},{13,13},{14, 8},{14,14},{16, 9},{16,10},{16,11},{16,12},{16,13},},
    };
    // clang-format on

    static constexpr auto hub_centers = array<Vec2<i8>, 9>{
        Vec2<i8>{6, 6}, {6, 16}, {6, 26}, {16, 6}, {16, 16}, {16, 26}, {26, 6}, {26, 16}, {26, 26},
    };

    auto hub_and_pet_pos_to_put_place = [&](const int& hub_id, const Vec2<i8>& pos) {
        const auto& hub_pos = hub_centers[hub_id];
        auto diff = pos - hub_pos;
        const auto n = SHORT_BAR ? 3 : 2;
        if (abs(diff.y) > abs(diff.x)) {
            diff.y = (diff.y > 0 ? n : -n);
            diff.x = (diff.x > 0 ? 1 : -1);
        } else {
            diff.y = (diff.y > 0 ? 1 : -1);
            diff.x = (diff.x > 0 ? n : -n);
        }
        return hub_pos + diff;
    };

    rep(idx_human, common::M) {
        const auto old_assigned_hub = human_states[idx_human].assigned_hub;
        human_states[idx_human].assigned_pet = -1;
        if ((int)human_states[idx_human].type >= (int)HumanState::Type::MOVING) {
            human_states[idx_human].moving_target_position = {-1, -1};
            human_states[idx_human].setting_target_position = {-1, -1};
        }
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

            // どこにも新しい行き場がないなら元の持ち場で待機
            if (human_states[idx_human].type == HumanState::Type::NONE && human_states[idx_human].assigned_hub == -1 && old_assigned_hub != -1) {
                human_states[idx_human].assigned_hub = old_assigned_hub;
                hub_assignments[old_assigned_hub] = idx_human;
                human_states[idx_human].moving_target_position = hub_centers[old_assigned_hub];
                if (features::distances_from_each_human[idx_human][hub_centers[old_assigned_hub]] <= (SHORT_BAR ? 3 : 2)) {
                    human_states[idx_human].type = HumanState::Type::WAITING;
                } else {
                    human_states[idx_human].type = HumanState::Type::MOVING;
                }
            }
        }
    }

    static auto caught = array<bool, 20>();

    // WAITING の人をペットに割り当てる
    struct PetCandidate {
        i8 pet;
        i8 cell;
        array<i8, 2> hubs;
        array<i8, 2> assigned_human;
        array<Vec2<i8>, 2> target_pos; // cell >= 0 なら置く場所、 cell < 0 ならペットの位置
        short dist;
        bool incomplete;
        inline bool operator<(const PetCandidate& rhs) const {
            if (incomplete != rhs.incomplete) {
                return incomplete < rhs.incomplete;
            }
            if ((cell < 0) != (rhs.cell < 0)) {
                return (cell < 0) < (rhs.cell < 0);
            }
            // if ((common::pet_types[pet] == PetType::DOG) != (common::pet_types[rhs.pet] == PetType::DOG)) {
            //     return (common::pet_types[pet] == PetType::DOG) < (common::pet_types[rhs.pet] == PetType::DOG);
            // }
            const auto n = SHORT_BAR ? 2 : 1;
            if (dist <= n || rhs.dist <= n && dist != rhs.dist) {
                return dist < rhs.dist;
            }
            if ((cell >= 24) != (rhs.cell >= 24)) {
                return (cell >= 24) < (rhs.cell >= 24);
            }
            return dist < rhs.dist;
        }
    };
    auto n_pet_candidates = 0;
    auto pet_candidates = array<PetCandidate, 20>();

    rep(i, common::N) {
        if (caught[i]) {
            continue;
        }
        auto cand = PetCandidate{};
        cand.pet = i;
        const auto& pet_pos = common::pet_positions[i];
        cand.cell = cell_ids[pet_pos];
        if (cand.cell == 99) // まだ壁をつくってない
            continue;
        if (cand.cell < 0) {
            const auto& hub = (i8)~cand.cell;
            cand.hubs = {hub, -1};
            const auto& assigned_human = hub_assignments[hub];
            if (assigned_human == -1 || human_states[assigned_human].type != HumanState::Type::WAITING)
                goto ng;
            cand.assigned_human = {assigned_human, -1};
            cand.target_pos = {pet_pos, {-1, -1}};
            cand.dist = max(features::distances_from_each_human[assigned_human][pet_pos] - 1, 0); // これでいいのか？
        } else if (cand.cell < 24) {
            const auto& hub = cell_id_to_hub_ids[cand.cell][0];
            cand.hubs = {hub, -1};
            const auto& assigned_human = hub_assignments[hub];
            if (assigned_human == -1 || human_states[assigned_human].type != HumanState::Type::WAITING)
                goto ng;
            cand.assigned_human = {assigned_human, -1};
            const auto& put_place = hub_and_pet_pos_to_put_place(hub, pet_pos);
            cand.target_pos = {put_place, {-1, -1}};
            cand.dist = abs(features::distances_from_each_human[assigned_human][put_place] - 1);
        } else {
            cand.hubs = cell_id_to_hub_ids[cand.cell];
            cand.dist = 0;
            auto ng_cnt = 0;
            rep(idx_hubs, 2) {
                const auto& hub = cand.hubs[idx_hubs];
                const auto& assigned_human = hub_assignments[hub];
                cand.assigned_human[idx_hubs] = assigned_human;
                if (assigned_human == -1 || human_states[assigned_human].type != HumanState::Type::WAITING) {
                    ng_cnt++;
                    cand.incomplete = true;
                }
                const auto& put_place = hub_and_pet_pos_to_put_place(hub, pet_pos);
                cand.target_pos[idx_hubs] = put_place;
                chmax(cand.dist, abs(features::distances_from_each_human[assigned_human][put_place] - 1));
            }
            if (ng_cnt == 2)
                goto ng;
        }
        pet_candidates[n_pet_candidates++] = cand;
    ng:;
    }

    // ペットに優先順位を付ける
    sort(pet_candidates.begin(), pet_candidates.begin() + n_pet_candidates);

    static auto cell_blocked = array<bool, 48>();
    auto check_connected = [&]() {
        auto uf = atcoder::dsu(9);
        rep3(i, 24, 48) {
            if (cell_blocked[i])
                continue;
            const auto& edge = cell_id_to_hub_ids[i];
            uf.merge(edge[0], edge[1]);
        }
        return uf.size(0) == 9;
    };

    // 優先度が高いペットから人を割り当て
    auto candidate_used = array<bool, 20>();
    rep(idx_cand, n_pet_candidates) {
        const auto& cand = pet_candidates[idx_cand];
        // if (cand.incomplete)
        //     continue;
        bool catching_condition = true;
        rep(idx_hubs, 2) {
            const auto& assigned_human = cand.assigned_human[idx_hubs];
            if (assigned_human == -1)
                continue;
            if (human_states[assigned_human].assigned_pet != -1 || human_states[assigned_human].type != HumanState::Type::WAITING)
                goto brcon;
        }
        candidate_used[idx_cand] = true;
        rep(idx_hubs, 2) {
            const auto& assigned_human = cand.assigned_human[idx_hubs];
            if (assigned_human == -1)
                continue;
            const auto& target_pos = cand.target_pos[idx_hubs];
            human_states[assigned_human].assigned_pet = cand.pet;
            human_states[assigned_human].setting_target_position = target_pos;
            if (cand.cell < 0)
                catching_condition = false;
            else if (features::distances_from_each_human[assigned_human][target_pos] != 1)
                catching_condition = false;
            else if (!Puttable(target_pos))
                catching_condition = false;
        }
        // 条件 (到着済み、壁が建設済み、設置可能、中に人が居ない) を確認して、状態を CATCHING に変更、caught の更新
        {
            // 両側に人が居るか？
            if (cand.incomplete)
                catching_condition = false;
            // 壁は建設済みか？
            for (const auto& fence_pos : cell_neighboring_fences[cand.cell]) {
                if (!common::fence_board[fence_pos])
                    catching_condition = false;
            }
            // 中に人は居ないか？
            rep(idx_human, common::M) {
                if (cell_ids[common::human_positions[idx_human]] == cand.cell)
                    catching_condition = false;
            }
            // 分断されないか？
            if (catching_condition && cand.cell >= 0) {
                assert(check_connected());
                cell_blocked[cand.cell] = true;
                if (!check_connected()) {
                    cell_blocked[cand.cell] = false;
                    catching_condition = false;
                }
            }
            // CATCHING にする
            if (catching_condition) {
                rep(idx_hubs, 2) {
                    const auto& assigned_human = cand.assigned_human[idx_hubs];
                    if (assigned_human == -1)
                        continue;
                    human_states[assigned_human].type = HumanState::Type::CATCHING;
                    caught[cand.pet] = true;
                }
                // 他のペットも同時に閉じ込められる場合
                rep(i, common::N) {
                    if (cell_ids[common::pet_positions[i]] == cand.cell)
                        caught[i] = true;
                }
            }
        }
    brcon:;
    }

    // すべて 4 なら、余ってる人を移動させる
    auto force_move = true;
    rep(idx_human, common::M) {
        if (human_states[idx_human].type != HumanState::Type::WAITING) {
            force_move = false;
        }
    }
    if (force_move) {
        auto neighboring_pet_counts = array<i8, 9>();
        rep(i, common::N) {
            // ペットが集まってる場所を調べる
            if (caught[i])
                continue;
            const auto& cell = cell_ids[common::pet_positions[i]];
            const auto& hubs = cell_id_to_hub_ids[cell];
            if (cell < 0) {
                neighboring_pet_counts[~cell]++;
            } else {
                neighboring_pet_counts[hubs[0]]++;
                if (cell >= 24) {
                    neighboring_pet_counts[hubs[1]]++;
                }
            }
        }
        auto best_human = -1;
        auto best_destination = -1;
        auto best = 0.25;
        rep(idx_human, common::M) {
            if (human_states[idx_human].assigned_pet != -1)
                continue;
            const auto& hub = human_states[idx_human].assigned_hub;
            if (hub == -1)
                continue; // そんなことある？
            const auto& current_n_pets = neighboring_pet_counts[hub];
            rep(new_hub, 9) {
                if (hub == new_hub)
                    continue;
                if (hub_assignments[new_hub] != -1)
                    continue;
                const auto& new_n_pets = neighboring_pet_counts[new_hub];
                const auto& dist = features::distances_from_each_human[idx_human][hub_centers[new_hub]];
                if (dist == 999)
                    continue;
                if (chmax(best, (double)(new_n_pets - current_n_pets) / (double)(dist + 1.0))) {
                    best_human = idx_human;
                    best_destination = new_hub;
                }
            }
        }
        if (best_human != -1) {
            hub_assignments[human_states[best_human].assigned_hub] = -1;
            hub_assignments[best_destination] = best_human;
            human_states[best_human].type = HumanState::Type::MOVING;
            human_states[best_human].moving_target_position = hub_centers[best_destination];
            human_states[best_human].assigned_hub = best_destination;
        }
    }

    // 片側だけでも集まる
    rep(idx_cand, n_pet_candidates) {
        if (candidate_used[idx_cand])
            continue;
        const auto& cand = pet_candidates[idx_cand];
        bool catching_condition = true;
        rep(idx_hubs, 2) {
            auto assigned_human = cand.assigned_human[idx_hubs];
            if (assigned_human == -1)
                continue;
            if (human_states[assigned_human].assigned_pet != -1 || human_states[assigned_human].type != HumanState::Type::WAITING)
                continue;
            const auto& target_pos = cand.target_pos[idx_hubs];
            human_states[assigned_human].assigned_pet = cand.pet;
            human_states[assigned_human].setting_target_position = target_pos;
        }
    }

    cout << "#";
    rep(i, common::N) { cout << caught[i]; }
    cout << endl;
    cout << "#assigned_pet=";
    rep(i, common::M) { cout << (int)human_states[i].assigned_pet << ","; }
    cout << endl;
    cout << "#assigned_hub=";
    rep(i, common::M) { cout << (int)human_states[i].assigned_hub << ","; }
    cout << endl;
    cout << "#moving_targetting_pos=";
    rep(i, common::M) { cout << (int)human_states[i].moving_target_position.y << "," << (int)human_states[i].moving_target_position.x << "|"; }
    cout << endl;
    cout << "#setting_targetting_pos=";
    rep(i, common::M) { cout << (int)human_states[i].setting_target_position.y << "," << (int)human_states[i].setting_target_position.x << "|"; }
    cout << endl;
    cout << "#type=";
    rep(i, common::M) { cout << (int)human_states[i].type << ","; }
    cout << endl;

    auto human_order = array<i8, 10>();
    iota(human_order.begin(), human_order.end(), 0);
    sort(human_order.begin(), human_order.begin() + common::M, [&](const i8& l, const i8& r) {
        return (human_states[l].type == HumanState::Type::SETTING) < (human_states[r].type == HumanState::Type::SETTING);
    }); // SETTING を最後にする

    rep(ord, common::M) {
        const auto& idx_human = human_order[ord];
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
            rep(i, 4) {
                // for (const auto& i : l_to_r ? array<int, 3>{0, 1, 2} : array<int, 3>{0, 1, 3}) {
                if (v != human_states[idx_human].setting_target_position) {
                    if (l_to_r && i == 3)
                        continue;
                    if (!l_to_r && i == 2)
                        continue;
                }
                const auto u = v + DIRECTION_VECS[i];
                if (pattern[u] && !common::fence_board[u]) {
                    n_remaining_put_place++;
                    if (TightPuttable(u)) {
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
                if (v == human_states[idx_human].setting_target_position) {
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
            if (distance_board[next_human_positions[idx_human]] == (SHORT_BAR ? 3 : 2)) {
                human_states[idx_human].type = HumanState::Type::WAITING;
            }
        } else if (human_states[idx_human].type == HumanState::Type::WAITING) {
            // ================================================ 4. 近接 ================================================
            // 既に良い位置にいるなら何もしない

            const auto& v = common::human_positions[idx_human];
            const auto& target_pos = human_states[idx_human].setting_target_position;
            if (target_pos.y == -1)
                continue; // 割り当てられなかった
            const auto& dist = features::distances_from_each_human[idx_human][target_pos];
            const auto& assigned_pet = human_states[idx_human].assigned_pet;
            auto distance_from_put_place = Board<short, 32, 32>();
            bfs(common::fence_board, target_pos, distance_from_put_place);
            if (dist >= 2) {
                // 置く場所に遠ければ近づく
                rep(i, 4) {
                    const auto u = v + DIRECTION_VECS[i];
                    if (distance_from_put_place[u] < dist) {
                        human_moves[idx_human] = "UDLR"[i];
                        next_human_positions[idx_human] = u;
                    }
                }
            } else if (features::distances_from_each_pet[assigned_pet][v] <= features::distances_from_each_pet[assigned_pet][target_pos]) {
                // ペットに近い側に居れば、中央に寄る
                auto distance_from_center = Board<short, 32, 32>();
                bfs(common::fence_board, hub_centers[human_states[idx_human].assigned_hub], distance_from_center);
                rep(i, 4) {
                    const auto u = v + DIRECTION_VECS[i];
                    if (distance_from_center[u] < distance_from_center[v]) {
                        human_moves[idx_human] = "UDLR"[i];
                        next_human_positions[idx_human] = u;
                    }
                }
            } // ちょうどいい位置に居れば何もしない
        } else if (human_states[idx_human].type == HumanState::Type::CATCHING) {
            // ================================================ 5. 捕獲 ================================================
            const auto& v = common::human_positions[idx_human];
            const auto& target_pos = human_states[idx_human].setting_target_position;
            rep(i, 4) {
                const auto u = v + DIRECTION_VECS[i];
                if (u == target_pos) {
                    human_moves[idx_human] = "udlr"[i];
                    next_human_positions[idx_human] = u;
                }
            }
            human_states[idx_human].type = HumanState::Type::WAITING;
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

void SetBarLength() {
    constexpr auto reg_long = RandomForest<20, 32>{
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 1, -2, -2, 2, -2, -2, 5, 6, -2, -2, 6, -2, -2, 6, 1, 5, -2, -2, 6, -2, -2, 2, 4, -2, -2, -2},
            {17.5, 13.5, 11.5, 4.5, -2.0, -2.0, 0.5, -2.0, -2.0, 5.5, 5.5, -2.0, -2.0, 6.5, -2.0,
             -2.0, 6.5,  3.5,  3.5, -2.0, -2.0, 5.5, -2.0, -2.0, 8.5, 3.5, -2.0, -2.0, -2.0},
            {53782868.063,       57099485.44150418,  60152927.10752688,  61610273.5879397,   61921321.31351351,  57500000.071428575,
             58476557.456647396, 50661616.09090909,  59007201.62345679,  53816594.401734106, 54299037.134969324, 51463342.44262295,
             54951781.950943395, 45952777.85,        35430555.5,         52967592.75,        45338430.90780142,  39048772.73255814,
             34204238.0,         43133796.53333333,  28623263.916666668, 43068705.80851064,  36697569.6,         47788065.96296296,
             48098178.88265306,  48283155.297435895, 50349881.82978723,  46359667.237623766, 12027778.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 4, 1, -2, -2, 3, -2, -2, 0, 2, -2, -2, 4, -2, -2, 0, 5, 6, -2, -2, 4, -2, -2, 6, 2, -2, -2, 4, -2, -2},
            {15.5, 11.5, 1.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 13.5, 0.5,  -2.0, -2.0, 0.5,  -2.0, -2.0,
             17.5, 1.5,  7.5, -2.0, -2.0, 7.5,  -2.0, -2.0, 6.5,  2.5,  -2.0, -2.0, 4.5,  -2.0, -2.0},
            {54682819.476,      59054595.97636364,  61442288.210045666, 62376623.37662338,  63434782.52173913,  61925925.96296296,
             60935641.67605634, 59970679.05555555,  61527777.82954545,  57474823.7734139,   58728219.69886363,  53700000.1,
             59031124.4939759,  56051612.91612903,  43138888.5,         56220406.69934641,  49339537.086666666, 52808754.236363634,
             46548850.68965517, 29892361.0,         52894180.0952381,   54143586.61029412,  54337654.333333336, 27944444.0,
             47331042.94736842, 41082967.973684214, 32954861.166666668, 44834401.884615384, 49603070.21052632,  50347141.96531792,
             46027392.05555555},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 6, 2, -2, -2, 5, -2, -2, 6, 2, -2, -2, 6, -2, -2, 6, 3, 5, -2, -2, 2, -2, -2, 0, 5, -2, -2, 2, -2, -2},
            {14.5, 11.5, 7.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 5.5, 0.5,  -2.0, -2.0, 6.5,  -2.0, -2.0,
             5.5,  3.5,  1.5, -2.0, -2.0, 8.0,  -2.0, -2.0, 18.5, 4.5, -2.0, -2.0, 8.5,  -2.0, -2.0},
            {53857097.225,       59195028.0210084,   61261710.24019608,  62119361.87128713,  60153333.28,       62766081.80263158,  60420712.03883495,
             56465811.961538464, 61756132.84415584,  57645016.356617644, 53709219.957446806, 29611111.0,        55950904.511627905, 58467160.49333333,
             56972839.48888889,  58840740.744444445, 49008137.1889313,   40508739.05617978,  33732165.38636363, 7458333.0,          35654640.92682927,
             47134722.2,         48132575.72727273,  3229167.0,          50747094.50804598,  52728020.92929293, 53823989.872727275, 49596681.09090909,
             46483796.34057971,  46735300.124087594, 12027778.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 1, 1, -2, -2, 2, -2, -2, 6, 5, -2, -2, 0, -2, -2, 6, 0, 4, -2, -2, 1, -2, -2, 2, 0, -2, -2, -2},
            {15.5, 11.5, 4.5,  0.5, -2.0, -2.0, 2.5, -2.0, -2.0, 5.5, 5.5,  -2.0, -2.0, 13.5, -2.0,
             -2.0, 5.5,  17.5, 2.5, -2.0, -2.0, 0.5, -2.0, -2.0, 8.5, 18.5, -2.0, -2.0, -2.0},
            {54331812.518,       58968559.223443225, 61350912.5410628,   61494333.92039801, 62799283.06451613, 61256372.605882354,
             56546296.333333336, 54319444.5,         61000000.0,         57513847.90560472, 54215277.72727273, 55142118.81395349,
             14361111.0,         58005838.0338983,   59048148.12592593,  57126388.89375,    48755460.75330397, 40402496.283783786,
             50284722.291666664, 37129629.666666664, 54669753.166666664, 35659027.8,        54444444.5,        34025513.30434783,
             50382090.67631579,  50585023.547619045, 52281600.440366976, 48273437.53125,    12027778.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 4, -2, -2, 2, -2, -2, 4, 2, -2, -2, 0, -2, -2, 6, 0, 1, -2, -2, 3, -2, -2, 2, 0, -2, -2, -2},
            {15.5, 13.5, 11.5, 1.5, -2.0, -2.0, 2.5, -2.0, -2.0, 0.5, 3.0,  -2.0, -2.0, 14.5, -2.0,
             -2.0, 5.5,  17.5, 2.5, -2.0, -2.0, 6.5, -2.0, -2.0, 8.5, 16.5, -2.0, -2.0, -2.0},
            {54434041.67,       58795037.2908778,   60142351.7100271,   60998516.72815534,  61929012.291666664, 60498548.96268657,
             59060327.20858896, 58347222.166666664, 59818565.48101266,  56449947.570754714, 51499999.8,         29944444.0,
             56888888.75,       56569511.52657005,  57468936.68817204,  55835769.94736842,  48386933.18377088,  38077506.390804596,
             47799768.44444445, 55666666.53333333,  42180555.52380952,  31214733.17647059,  27430713.431818184, 55000000.14285714,
             51088499.84337349, 51206508.06646526,  54345299.092307694, 50439510.2593985,   12027778.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 6, 0, -2, -2, 1, -2, -2, 4, 6, -2, -2, 1, -2, -2, 2, 0, 5, -2, -2, 6, -2, -2, 1, 5, -2, -2, -2},
            {14.5, 12.5, 7.5,  10.5, -2.0, -2.0, 2.5, -2.0, -2.0, 0.5, 6.5, -2.0, -2.0, 2.5, -2.0,
             -2.0, 8.5,  16.5, 5.5,  -2.0, -2.0, 6.5, -2.0, -2.0, 3.5, 2.5, -2.0, -2.0, -2.0},
            {54048826.379,      59899936.86363637,  61129393.945454545, 62013396.36879433, 63238993.69811321,  61275252.52272727,
             60199212.29104478, 60840628.5050505,   58384920.71428572,  57850841.72727273, 40074073.666666664, 29944444.0,
             60333333.0,        58180041.13580247,  58831908.85897436,  57574735.39285714, 49451525.28392857,  49809959.944144145,
             53447435.86666667, 54371393.464088395, 41501984.071428575, 47839660.48611111, 42883136.93939394,  49719721.14176245,
             9665278.0,         5428819.75,         3229167.0,          12027778.0,        26611111.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 4, -2, -2, 2, -2, -2, 5, 0, -2, -2, 6, -2, -2, 6, 0, 4, -2, -2, 1, -2, -2, 4, 0, -2, -2, 1, -2, -2},
            {15.5, 11.5, 10.5, 3.5,  -2.0, -2.0, 0.5,  -2.0, -2.0, 5.5,  13.5, -2.0, -2.0, 5.5,  -2.0, -2.0,
             5.5,  17.5, 5.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 3.5,  16.5, -2.0, -2.0, 5.5,  -2.0, -2.0},
            {54387798.614,      58925383.136206895, 61908888.88,        62456349.196428575, 62678486.957446806, 61296296.44444445,
             61212121.20454545, 63222222.2,         61091030.78313253,  57355116.95526316,  57594594.5945946,   58664431.67816092,
             56644841.26530612, 48494444.3,         14361111.0,         57027777.625,       48121610.46428572,  41143644.32608695,
             51627688.09677419, 53986111.071428575, 29615740.333333332, 35815687.655737706, 40335016.78787879,  30489335.464285713,
             50078844.86890244, 51938720.551515155, 55068965.62068965,  51271241.823529415, 48196148.62576687,  49179054.074324325,
             38498148.2},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 4, 0, -2, -2, 2, -2, -2, 6, 5, -2, -2, 1, -2, -2, 6, 0, 4, -2, -2, 5, -2, -2, 0, 3, -2, -2, 6, -2, -2},
            {15.5, 13.5, 1.5, 11.5, -2.0, -2.0, 1.5,  -2.0, -2.0, 5.5, 5.5,  -2.0, -2.0, 6.5,  -2.0, -2.0,
             5.5,  17.5, 2.5, -2.0, -2.0, 2.5,  -2.0, -2.0, 17.5, 0.5, -2.0, -2.0, 6.5,  -2.0, -2.0},
            {54103541.672,      58725280.77978339, 60197453.697222225, 61637681.15652174,  62335917.31395349,  59567049.79310345, 59521428.56326531,
             60845751.61176471, 58817881.94375,    55993413.51030928,  51068518.5,         53690476.178571425, 14361111.0,        56894308.93902439,
             56950310.55279503, 53888889.0,        48362637.040358745, 41472130.828947365, 49559116.743589744, 41575925.93333333, 54548611.0,
             32948010.54054054, 20167534.75,       36473659.03448276,  49777984.262162164, 52138532.788461536, 37211111.2,        52632818.27152318,
             48057210.57009346, 43028409.20454545, 49358782.6882353},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 1, 4, -2, -2, 6, -2, -2, 0, 2, -2, -2, 6, -2, -2, 6, 0, 4, -2, -2, 3, -2, -2, 2, 0, -2, -2, -2},
            {15.5, 11.5, 2.5,  3.5, -2.0, -2.0, 7.5, -2.0, -2.0, 14.5, 0.5,  -2.0, -2.0, 6.5, -2.0,
             -2.0, 5.5,  17.5, 5.5, -2.0, -2.0, 5.5, -2.0, -2.0, 8.5,  17.5, -2.0, -2.0, -2.0},
            {53378180.563,       58681498.94716981,  61496880.59893048,  62064734.36521739,  62226495.79807692,  60535353.54545455,
             60589891.94444445,  62038888.825,       58778645.84375,     57146582.41982507,  58168880.02390438,  54341880.23076923,
             58377917.827731095, 54357487.86956522,  50984375.0,         56156481.4,         47397842.8106383,   37659313.705882356,
             47222222.166666664, 51763888.884615384, 17701388.5,         32443181.818181816, 29684692.702127658, 48649305.375,
             49547907.67792208,  49842568.38219895,  51946921.439490445, 48374197.58222222,  12027778.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 3, -2, -2, 2, -2, -2, 5, 5, -2, -2, 2, -2, -2, 6, 1, 0, -2, -2, 5, -2, -2, 0, 0, -2, -2, 2, -2, -2},
            {15.5, 14.5, 11.5, 1.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 5.5,  3.5,  -2.0, -2.0, 1.5,  -2.0, -2.0,
             5.5,  3.5,  17.5, -2.0, -2.0, 1.5,  -2.0, -2.0, 17.5, 16.5, -2.0, -2.0, 8.5,  -2.0, -2.0},
            {54345027.783,      58844750.89323843,  59605413.126068376, 60955095.031088084, 58638888.96875,     62104220.51937985,
             58658181.82545455, 57966490.246031746, 59243102.22147651,  55057624.0319149,   56364705.8117647,   57252991.384615384,
             53477777.7,        42712962.777777776, 31372222.0,         56888888.75,        48571410.45890411,  41369309.347222224,
             46724073.91111111, 53842995.04347826,  39281565.45454545,  32444701.740740743, 16347222.333333334, 37043981.571428575,
             49988217.23497268, 53097222.24242424,  54490610.40845071,  51475409.78688525,  48234419.538461536, 48389812.84978541,
             12027778.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 4, -2, -2, 5, -2, -2, 5, 0, -2, -2, 6, -2, -2, 6, 0, 5, -2, -2, 1, -2, -2, 4, 5, -2, -2, 1, -2, -2},
            {16.5, 13.5, 11.5, 1.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 5.5, 15.5, -2.0, -2.0, 6.5,  -2.0, -2.0,
             6.5,  17.5, 1.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 3.5,  1.5, -2.0, -2.0, 5.5,  -2.0, -2.0},
            {53916548.622,       58036864.781395346, 59932268.88976378,  61242803.71502591,  62076023.368421055, 60701567.017094016,
             58586879.41489362,  59769230.8021978,   57477663.164948456, 55301452.03409091,  55602733.702380955, 56651084.012195125,
             53648989.94318182,  48974537.0,         31979166.5,         57472222.25,        46430340.388732396, 38987816.815789476,
             47986111.083333336, 27694444.5,         52044444.4,         36588271.677777775, 28770833.285714287, 40118727.72580645,
             49950870.21161826,  51609733.66942149,  45759259.0,         51914975.82608695,  48278182.891666666, 49080017.54054054,
             38388888.88888889},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 6, -2, -2, 2, -2, -2, 2, 6, -2, -2, 3, -2, -2, 6, 5, 5, -2, -2, 3, -2, -2, 0, 4, -2, -2, 1, -2, -2},
            {15.5, 13.5, 11.5, 8.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 5.5, 5.5,  -2.0, -2.0, 3.5,  -2.0, -2.0,
             6.5,  3.5,  1.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 19.5, 2.5, -2.0, -2.0, 2.5,  -2.0, -2.0},
            {53289687.509,       58819362.250493094, 59999598.56358381,  61007725.142011836, 62360269.333333336, 59094841.21428572,
             59037037.028248586, 58098108.680851065, 60100401.6626506,   56282953.77639752,  56603174.61038961,  51750000.0,
             57165861.52173913,  49238095.428571425, 57666666.8,         28166667.0,         47602983.46450304,  42711576.86227545,
             46704687.5,         39175000.0,         50127272.72727273,  39039750.98850574,  34272777.82,        45481606.62162162,
             50108704.02453988,  51170440.30188679,  53933895.974683546, 49996714.50537635,  45496243.14754099,  36176587.21428572,
             48272310.872340426},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 1, -2, -2, 5, -2, -2, 2, 5, -2, -2, 3, -2, -2, 6, 0, 1, -2, -2, 1, -2, -2, 2, 0, -2, -2, -2},
            {15.5, 14.5, 11.5, 4.5, -2.0, -2.0, 4.5, -2.0, -2.0, 5.5, 5.5,  -2.0, -2.0, 3.0, -2.0,
             -2.0, 5.5,  17.5, 3.5, -2.0, -2.0, 5.5, -2.0, -2.0, 8.5, 18.5, -2.0, -2.0, -2.0},
            {54063756.943,      58529289.13320825,  59398943.03016242,  60759462.75824176,  61199539.77514793,  55038461.538461536,
             58404506.92369478, 58675213.683257915, 56267857.14285714,  54854575.11764706,  55396745.171717174, 56176543.14444444,
             47598765.44444445, 36962963.333333336, 54555556.0,         28166667.0,         48967121.70235546,  40822656.2375,
             49624249.18918919, 52437499.89285714,  40871913.666666664, 33249192.534883723, 34758854.175,       13120370.666666666,
             50650732.13436692, 50851370.85714286,  52416284.42660551,  48808549.5508982,   12027778.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 6, -2, -2, 1, -2, -2, 5, 0, -2, -2, 4, -2, -2, 6, 0, 5, -2, -2, 4, -2, -2, 4, 0, -2, -2, 6, -2, -2},
            {15.5, 11.5, 10.5, 9.5,  -2.0, -2.0, 4.5,  -2.0, -2.0, 4.5,  13.5, -2.0, -2.0, 4.5,  -2.0, -2.0,
             5.5,  18.5, 1.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 7.5,  17.5, -2.0, -2.0, 7.0,  -2.0, -2.0},
            {53900152.785,       58780723.122486286, 61819177.33173077, 62549655.84955752,  62738173.8019802,  60962963.083333336, 60950292.35789473,
             61159800.21348315,  57842592.5,         56916420.83480826, 57477331.83946488,  58771739.13043478, 56367839.875776395, 52723611.075,
             55452160.416666664, 28166667.0,         48006837.16777042, 38422962.986666664, 46135732.29545455, 32422839.555555556, 49661904.71428572,
             27475806.548387095, 20074652.9375,      35370370.4,        49908399.50529101,  50168869.51344086, 52490038.33793104,  48686184.581497796,
             33759259.0,         25694444.0,         49888889.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 2, -2, -2, 1, -2, -2, 6, 5, -2, -2, 0, -2, -2, 2, 6, 0, -2, -2, 0, -2, -2, 1, 6, -2, -2, -2},
            {15.5, 11.5, 10.5, 1.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 5.5, 5.5, -2.0, -2.0, 14.5, -2.0,
             -2.0, 8.5,  5.5,  17.5, -2.0, -2.0, 19.5, -2.0, -2.0, 3.5, 6.0, -2.0, -2.0, -2.0},
            {53946652.796,       58947978.894265234, 61781407.005025126, 62470899.40952381,  61722222.125,      63101364.491228074,
             61011229.319148935, 61353276.35897436,  59343750.0,         57377359.96935933,  53893518.53703704, 54639413.018867925,
             14361111.0,         57994171.23934426,  58767094.02884615,  56336769.793814436, 47632761.47737557, 48383021.51388889,
             39098689.32394366,  48647849.41935484,  31698090.25,        50209025.90581717,  51336629.06101695, 45168981.5,
             15221527.9,         7628472.5,          3229167.0,          12027778.0,         26611111.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, -1, 27, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 26, -1, 28, -1, -1},
            {0, 0, 0, 1, -2, -2, 5, -2, -2, 5, 6, -2, -2, 3, -2, -2, 2, 6, 3, -2, -2, 0, -2, -2, 5, -2, 3, -2, -2},
            {15.5, 13.5, 11.5, 4.5, -2.0, -2.0, 4.5,  -2.0, -2.0, 5.5, 5.5,  -2.0, -2.0, 4.5, -2.0,
             -2.0, 8.5,  5.5,  5.5, -2.0, -2.0, 16.5, -2.0, -2.0, 1.5, -2.0, 4.5,  -2.0, -2.0},
            {54674944.448,      58691276.80350877,  60103698.716981135, 61119144.89215686,  61391880.35897436,  55209876.44444445,
             58863273.4491018,  59324757.63758389,  55043209.88888889,  56058068.110552765, 56418003.56684492,  53664021.0952381,
             56766398.93975904, 50449073.916666664, 57617283.88888889,  28944444.0,         49350969.0,         49742140.52941176,
             44839041.04109589, 42075617.277777776, 52692982.2631579,   50758976.50284091,  54858585.836363636, 49999789.58922559,
             16101389.0,        26611111.0,         9094907.666666666,  12027778.0,         3229167.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 1, -2, -2, 0, -2, -2, 0, 5, -2, -2, 4, -2, -2, 6, 5, 3, -2, -2, 3, -2, -2, 2, 6, -2, -2, -2},
            {17.5, 13.5, 11.5, 4.5, -2.0, -2.0, 12.5, -2.0, -2.0, 15.5, 5.5, -2.0, -2.0, 7.5, -2.0,
             -2.0, 5.5,  2.5,  6.5, -2.0, -2.0, 3.5,  -2.0, -2.0, 8.5,  6.5, -2.0, -2.0, -2.0},
            {54395423.628,       57761118.71311475,  60472941.87305699,  61695360.23076923, 61877075.38505747, 57743055.625,
             59382352.946078435, 60035353.59090909,  58886973.14655172,  54735789.98554913, 56102541.34042553, 56549999.97777778,
             46034722.0,         53109528.87974683,  53276164.922580644, 44500000.0,        45202554.96268657, 34200828.87096774,
             26833664.04761905,  20415441.23529412,  54111111.0,         37974254.75609756, 29691288.0,        47565058.368421055,
             48513754.0776699,   48691734.448780484, 43197447.51351351,  49901785.73809524, 12027778.0},
        },
        DecisionTree<32>{
            29,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, -1},
            {0, 0, 0, 3, -2, -2, 5, -2, -2, 0, 6, -2, -2, 6, -2, -2, 6, 0, 4, -2, -2, 2, -2, -2, 2, 4, -2, -2, -2},
            {15.5, 13.5, 11.5, 1.5, -2.0, -2.0, 4.5, -2.0, -2.0, 14.5, 5.5, -2.0, -2.0, 5.5, -2.0,
             -2.0, 5.5,  17.5, 5.5, -2.0, -2.0, 6.5, -2.0, -2.0, 8.5,  2.5, -2.0, -2.0, -2.0},
            {54390638.893,       58843731.17540687, 60054236.50837989, 60813117.255555555, 58768832.423728816, 61809917.29752066,
             59286829.01123595,  59497671.38323353, 56085858.45454545, 56621367.538461536, 57675830.5257732,   59051282.0,
             57462963.03571428,  55577664.37755102, 51568627.35294118, 56419067.209876545, 48881556.04697987,  36516865.03174603,
             44129830.82608695,  45796717.09090909, 7458333.0,         32139409.7,         29528390.5,         46935185.166666664,
             50910138.166666664, 51011658.68929504, 53247474.77272727, 50110780.41391941,  12027778.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 1, 2, -2, -2, 4, -2, -2, 0, 1, -2, -2, 4, -2, -2, 6, 3, 4, -2, -2, 4, -2, -2, 2, 0, -2, -2, 3, -2, -2},
            {15.5, 11.5, 4.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 13.5, 0.5,  -2.0, -2.0, 0.5,  -2.0, -2.0,
             5.5,  3.5,  4.5, -2.0, -2.0, 5.5,  -2.0, -2.0, 7.5,  16.5, -2.0, -2.0, 3.5,  -2.0, -2.0},
            {53904076.397,       58772109.887867644, 61559405.975247525, 61695462.518324606, 60849616.89655172,  62064327.526315786,
             59196969.63636363,  62166666.5,         51277778.0,         57125812.19883041,  58433845.03947368,  53234568.0,
             58761072.26573426,  56079385.92631579,  38740740.333333336, 56357546.016042784, 48096597.846491225, 40879827.280487806,
             33948578.14285714,  27781703.04347826,  41413742.7368421,   48157638.875,       50293836.78125,     39612847.25,
             49678884.441176474, 49950294.38858695,  53951690.79710145,  49026895.217391305, 33032407.666666668, 54000000.0,
             22548611.5},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 0, 0, 2, -2, -2, 1, -2, -2, 4, 0, -2, -2, 1, -2, -2, 6, 0, 1, -2, -2, 2, -2, -2, 4, 0, -2, -2, 2, -2, -2},
            {16.5, 11.5, 10.5, 2.5,  -2.0, -2.0, 4.5,  -2.0, -2.0, 7.5,  14.5, -2.0, -2.0, 2.0,  -2.0, -2.0,
             5.5,  17.5, 4.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 4.5,  19.5, -2.0, -2.0, 6.5,  -2.0, -2.0},
            {54049937.482,       58084191.49105691,  61781028.35106383, 62484358.097087376, 62034074.013333336, 63690476.178571425,
             60928758.1882353,   61244170.11111111,  54541666.75,       56456544.34894614,  56603904.59669811,  57777777.75757576,
             54667013.88125,     35629629.333333336, 27944444.0,        51000000.0,         47605609.64935065,  38998532.83098592,
             53868421.05263158,  55274509.823529415, 41916666.5,        33565304.442307696, 25839975.826086957, 39692289.20689655,
             49551795.808917195, 50706803.71806167,  51398745.51612903, 47567750.682926826, 46538154.48275862,  47564814.76190476,
             17791666.666666668},
        },
    };
    constexpr auto reg_short = RandomForest<20, 32>{
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 1, -2, -2, 5, -2, -2, 0, 6, -2, -2, 4, -2, -2, 6, 5, 3, -2, -2, 1, -2, -2, 0, 6, -2, -2, 6, -2, -2},
            {15.5, 5.5, 13.5, 0.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 11.5, 7.5,  -2.0, -2.0, 4.5,  -2.0, -2.0,
             5.5,  1.5, 3.5,  -2.0, -2.0, 5.5,  -2.0, -2.0, 17.5, 6.5,  -2.0, -2.0, 7.5,  -2.0, -2.0},
            {53156263.871,       59993792.42805755,  49054038.88505747, 55857212.456140354, 33777777.5,         57523584.905660376, 36128009.1,
             44115530.13636363,  14162326.25,        62023128.37313433, 65322619.13142857,  63779671.78787879,  66256880.82568807,  60059145.77891157,
             61339467.49583333,  54368827.03703704,  44593953.33558559, 24298409.76811594,  50169191.72727273,  27854166.25,        62920634.85714286,
             19391882.155172415, 21411815.725490198, 4675223.285714285, 48328333.352,       54643500.870229006, 43555871.18181818,  56881371.08256881,
             44937813.086065575, 39036938.4893617,   48635694.5},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 4, -2, -2, 4, -2, -2, 0, 4, -2, -2, 5, -2, -2, 6, 5, 3, -2, -2, 2, -2, -2, 0, 1, -2, -2, 6, -2, -2},
            {15.5, 5.5, 1.5, 4.5,  -2.0, -2.0, 3.5,  -2.0, -2.0, 13.5, 3.5,  -2.0, -2.0, 5.5,  -2.0, -2.0,
             5.5,  1.5, 3.5, -2.0, -2.0, 2.5,  -2.0, -2.0, 17.5, 3.5,  -2.0, -2.0, 7.5,  -2.0, -2.0},
            {52520708.33,       59952979.78363636,  49778673.817204304, 61755952.35714286,  62841563.777777776, 32444444.0,
             44619230.75384615, 46242383.48387097,  11074074.333333334, 62023462.17943107,  64067973.37071651,  64895252.362549804,
             61101587.27142857, 57197814.441176474, 58289062.40625,     39737847.0,         43436820.997777775, 15582209.955882354,
             38555555.375,      21037036.666666668, 49066666.6,         12519097.233333332, 4802997.105263158,  16094850.951219512,
             48395233.43455497, 55174968.69172932,  52570815.078947365, 58647173.508771926, 44773929.06024096,  38511574.03333333,
             48318658.320754714},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 0, 5, 0, -2, -2, 1, -2, -2, 5, 3, -2, -2, 3, -2, -2, 0, 6, 5, -2, -2, 0, -2, -2, 6, 5, -2, -2, 0, -2, -2},
            {5.5,  14.5, 1.5, 12.5, -2.0, -2.0, 1.5,  -2.0, -2.0, 1.5, 3.5,  -2.0, -2.0, 1.5,  -2.0, -2.0,
             15.5, 7.5,  3.5, -2.0, -2.0, 12.5, -2.0, -2.0, 6.5,  4.5, -2.0, -2.0, 17.5, -2.0, -2.0},
            {52039358.498,      33212752.036144577, 49801948.051948056, 67059259.333333336, 68131313.27272727, 64111111.0,        45626792.09677419,
             30737847.125,      50805555.56521739,  18860301.550561797, 40191666.5,         15583333.0,        50738095.14285714, 16160128.772151899,
             62333333.0,        15568164.615384616, 55786620.695443645, 61238331.03278688,  58255416.645,      60256054.12179487, 51162247.40909091,
             63309799.35763889, 66305555.59090909,  60774928.69871795,  48097503.22543353,  36923892.60810811, 40753844.21428572, 25008487.611111112,
             51137382.58455882, 55971982.801724136, 47542423.44871795},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 3, -2, -2, 5, -2, -2, 0, 6, -2, -2, 1, -2, -2, 6, 5, 3, -2, -2, 0, -2, -2, 6, 0, -2, -2, 4, -2, -2},
            {15.5, 5.5, 12.5, 2.5,  -2.0, -2.0, 4.5,  -2.0, -2.0, 13.5, 7.5,  -2.0, -2.0, 3.5,  -2.0, -2.0,
             5.5,  1.5, 6.5,  -2.0, -2.0, 18.5, -2.0, -2.0, 7.5,  18.5, -2.0, -2.0, 6.5,  -2.0, -2.0},
            {52129761.283,       59229242.95604396,  45951103.47945205, 54191137.64285714,  48903769.928571425, 64765873.071428575,
             34787186.22580645,  39188271.481481485, 5079860.75,        61278511.83932347,  63633919.07028754,  61422960.046875,
             65163663.691891894, 56670746.44375,     60087709.26027397, 53803639.712643676, 43591618.12555066,  21930051.094594594,
             53476190.28571428,  41259259.0,         62638888.75,       18634185.80597015,  23569608.594594594, 12547164.366666667,
             47809923.284210525, 43165027.59006211,  47018162.375,      36134746.578947365, 51224663.9543379,   52422202.16908213,
             30567129.75},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 5, 0, 0, -2, -2, 3, -2, -2, 0, 2, -2, -2, 2, -2, -2, 0, 0, 0, -2, -2, 6, -2, -2, 6, 4, -2, -2, 3, -2, -2},
            {5.5,  1.5,  11.5, 10.5, -2.0, -2.0, 2.5,  -2.0, -2.0, 12.5, 2.5,  -2.0, -2.0, 6.5,  -2.0, -2.0,
             17.5, 13.5, 11.5, -2.0, -2.0, 7.5,  -2.0, -2.0, 7.5,  1.5,  -2.0, -2.0, 5.5,  -2.0, -2.0},
            {52470873.243,       34259939.47953217,  56907163.65789474,  68197530.8888889,  69133333.4,         67027777.75,
             53403256.5862069,   32899999.8,         57674768.416666664, 27789304.0,        51427579.46428572,  56054293.09090909,
             34462962.833333336, 21485763.876190476, 20020329.85,        50794444.4,        56227290.21954162,  59903555.443217665,
             63549422.78896104,  65278251.35227273,  61244318.03787879,  56458993.65644172, 52454006.37692308,  59115362.77040816,
             44274715.082051285, 40071454.618421055, 52680555.5,         38160984.78787879, 46959150.336134456, 45248638.34313726,
             57222222.294117644},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 1, -2, -2, 0, -2, -2, 0, 3, -2, -2, 6, -2, -2, 6, 5, 3, -2, -2, 2, -2, -2, 0, 6, -2, -2, 5, -2, -2},
            {13.5, 5.5, 1.5, 2.5,  -2.0, -2.0, 11.5, -2.0, -2.0, 10.5, 3.5,  -2.0, -2.0, 7.5,  -2.0, -2.0,
             5.5,  2.5, 2.5, -2.0, -2.0, 3.5,  -2.0, -2.0, 17.5, 6.5,  -2.0, -2.0, 1.5,  -2.0, -2.0},
            {52201258.655,      61565415.889502764, 51610714.3,         62384920.60714286,  54648148.0,        68187500.0625,
             44427910.0952381,  53619883.10526316,  36834541.08695652,  63951816.95547945,  67569069.21621622, 67948853.76190476,
             65393939.54545455, 62723942.33486239,  59866873.44680851,  64889784.87903226,  46888053.45297806, 28553938.539215688,
             48929067.39285714, 16684027.75,        54303240.666666664, 20844430.324324325, 25669687.44,       10791811.333333334,
             50377008.15671642, 54633075.413680784, 45824448.455882356, 57139295.635983266, 44671276.069869,   54612318.82608695,
             43561353.82038835},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 0, 2, 5, -2, -2, 5, -2, -2, 5, 1, -2, -2, 5, -2, -2, 0, 0, 0, -2, -2, 5, -2, -2, 6, 4, -2, -2, 0, -2, -2},
            {5.5,  14.5, 0.5,  2.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 1.5, 2.5,  -2.0, -2.0, 5.5,  -2.0, -2.0,
             15.5, 12.5, 10.5, -2.0, -2.0, 6.5,  -2.0, -2.0, 7.5,  1.5, -2.0, -2.0, 19.5, -2.0, -2.0},
            {51697852.414,       32585097.72826087, 47031035.67901234,  65086419.777777776, 63222222.0,         67416667.0,        44774112.666666664,
             50417874.5,         34788995.57692308, 21224699.922330096, 46482323.18181818,  62296296.166666664, 27505555.6,        18204766.70652174,
             20283821.139240507, 5570512.846153846, 56007591.21568628,  61544142.733606555, 64630163.86752137,  66719883.18947368, 63201938.43165468,
             58701115.38976378,  58901020.98418973, 8125000.0,          47770282.8597561,   41461033.92361111,  59371345.15789474, 38738666.616,
             52707955.94021739,  54530244.92763158, 44052083.25},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 5, -2, -2, 4, -2, -2, 0, 2, -2, -2, 6, -2, -2, 6, 5, 4, -2, -2, 4, -2, -2, 0, 3, -2, -2, 6, -2, -2},
            {15.5, 5.5, 4.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 11.5, 0.5,  -2.0, -2.0, 7.5,  -2.0, -2.0,
             6.5,  1.5, 2.5, -2.0, -2.0, 5.5,  -2.0, -2.0, 17.5, 0.5,  -2.0, -2.0, 8.5,  -2.0, -2.0},
            {51923582.433,       59280648.261732854, 47364541.48192771,  51004111.85526316,  62207070.77272727,  46439943.40740741, 7849206.0,
             31944444.0,         3833333.0,          61380514.212314226, 64985388.612716764, 54162393.15384615,  65864756.99375,    59287751.55704698,
             55724190.59055118,  61934372.85964912,  42784984.96860986,  28208373.826666668, 42320833.4,         63500000.0,        37026041.75,
             26037226.2,         28019247.163636364, 15136110.9,         50171781.1554054,   56241291.865079366, 27930555.4,        57411157.00826446,
             45673202.629411764, 41120077.35443038,  49625915.78021978},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 3, -2, -2, 1, -2, -2, 0, 4, -2, -2, 4, -2, -2, 6, 6, 5, -2, -2, 1, -2, -2, 0, 4, -2, -2, 6, -2, -2},
            {15.5, 5.5, 12.5, 2.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 11.5, 1.5,  -2.0, -2.0, 6.5,  -2.0, -2.0,
             6.5,  5.5, 0.5,  -2.0, -2.0, 4.5,  -2.0, -2.0, 17.5, 6.5,  -2.0, -2.0, 7.5,  -2.0, -2.0},
            {52436314.218,       59855575.18301887, 48154442.52054795,  54363636.43181818,  49545138.90625,     67212963.16666667,
             38733596.5862069,   28182802.17647059, 53680555.333333336, 61724683.901531726, 65510969.66878981,  67798611.21875,
             63936678.70967742,  59743194.35,       60001360.64527027,  40638888.5,         44069913.555319145, 30908347.76190476,
             24083312.847058825, 64111111.0,        23118787.590361446, 37897841.34939759,  33330577.53968254,  52284722.35,
             51391579.294701986, 58435648.13333333, 59167641.315789476, 44527777.666666664, 46747138.3021978,   40368480.71428572,
             49097170.04511278},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 0, -2, -2, 4, -2, -2, 0, 0, -2, -2, 2, -2, -2, 6, 6, 5, -2, -2, 5, -2, -2, 0, 1, -2, -2, 1, -2, -2},
            {15.5, 5.5, 4.5, 12.5, -2.0, -2.0, 1.5,  -2.0, -2.0, 13.5, 11.5, -2.0, -2.0, 0.5,  -2.0, -2.0,
             6.5,  5.5, 3.5, -2.0, -2.0, 4.5,  -2.0, -2.0, 17.5, 0.5,  -2.0, -2.0, 5.5,  -2.0, -2.0},
            {52777683.147,      60118061.70106762, 48410001.63218391,  51679834.12987013,  57146135.36956522,  43568548.41935484, 23232291.4,
             41347222.0,        11155671.0,        62262485.33473684,  64328223.251572326, 66003133.548780486, 62544552.54545455, 58078379.23566879,
             47920454.36363637, 58843702.34246575, 43359206.55479452,  28817248.060402684, 20160095.944444444, 28555972.16,       15694204.340425532,
             36912247.44155844, 39650534.15384615, 22079861.083333332, 50856617.6816609,   56889698.0,         15194444.0,        58140555.62,
             47515718.3655914,  48674039.81481481, 39697048.583333336},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 0, -2, -2, 5, -2, -2, 0, 0, -2, -2, 6, -2, -2, 6, 5, 2, -2, -2, 6, -2, -2, 3, 0, -2, -2, 5, -2, -2},
            {17.5, 5.5, 14.5, 12.5, -2.0, -2.0, 3.5,  -2.0, -2.0, 15.5, 11.5, -2.0, -2.0, 6.5,  -2.0, -2.0,
             6.5,  2.5, 4.5,  -2.0, -2.0, 5.5,  -2.0, -2.0, 3.5,  18.5, -2.0, -2.0, 4.5,  -2.0, -2.0},
            {52685920.139,       57442703.65725806, 44352784.15740741,  51315674.6,         55154100.5952381,   45558035.60714286,  31526407.02631579,
             41693236.56521739,  15937268.4,        59665520.17610063,  61744922.97058824,  64806054.670658685, 60090524.932038836, 53479296.8625,
             39789457.09090909,  55661735.08695652, 38861518.0390625,   24985956.777777776, 35157118.0,         27588235.17647059,  43735185.2,
             19374281.620689657, 14402126.8125,     25493856.769230768, 46384412.69879518,  49650745.317073174, 56410493.88888889,  47749566.03125,
             43195849.9047619,   46791556.52380952, 32408730.04761905},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 5, 0, 3, -2, -2, 5, -2, -2, 0, 2, -2, -2, 0, -2, -2, 0, 0, 4, -2, -2, 3, -2, -2, 6, 5, -2, -2, 0, -2, -2},
            {5.5,  2.5,  17.5, 2.5,  -2.0, -2.0, 0.5,  -2.0, -2.0, 17.5, 3.5,  -2.0, -2.0, 19.5, -2.0, -2.0,
             15.5, 11.5, 2.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 6.5,  4.5,  -2.0, -2.0, 17.5, -2.0, -2.0},
            {51824186.606,       33489557.293413173, 51075559.67164179, 54321958.75862069,  49250257.18518519, 58739247.22580645, 30154321.111111112,
             58777778.0,         26576389.0,         21706935.7,        28610048.362068966, 32794482.48780488, 18518177.82352941, 12174065.833333334,
             14549938.705882354, 2076606.125,        55499916.6122449,  62139642.25,        66662640.97101449, 67572549.09411764, 65203354.358490564,
             59957216.29370629,  58141128.96129032,  62106021.91603053, 48616680.25427873,  40723309.86746988, 44321828.32835821, 25654513.8125,
             50626342.0398773,   54679789.27586207,  47379105.3038674},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 0, 5, 0, -2, -2, 2, -2, -2, 5, 1, -2, -2, 5, -2, -2, 0, 0, 0, -2, -2, 2, -2, -2, 6, 3, -2, -2, 0, -2, -2},
            {5.5,  14.5, 0.5,  10.5, -2.0, -2.0, 0.5,  -2.0, -2.0, 1.5, 2.5,  -2.0, -2.0, 5.5,  -2.0, -2.0,
             15.5, 13.5, 10.5, -2.0, -2.0, 0.5,  -2.0, -2.0, 6.5,  6.5, -2.0, -2.0, 17.5, -2.0, -2.0},
            {52782589.408,      34431947.96644295,  50070810.67213115,  67791666.875,       69722222.5,         65861111.25,       47395964.45283019,
             64537037.0,        45207742.42553192,  23591372.681818184, 45013888.78571428,  61388888.75,        38463888.8,        19538464.22972973,
             21371622.6875,     7806250.1,          55995568.93184489,  61662446.08405172,  63700549.946308725, 67129861.2375,     62442087.08715596,
             58003681.31927711, 45350308.61111111,  59542605.027027026, 49201173.586563304, 38447696.848101266, 35917546.91549296, 60902777.5,
             51959370.54220779, 57549164.274336286, 48720156.73846154},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 2, -2, -2, 0, -2, -2, 0, 5, -2, -2, 4, -2, -2, 6, 5, 2, -2, -2, 0, -2, -2, 6, 3, -2, -2, 5, -2, -2},
            {15.5, 5.5, 1.5, 2.5,  -2.0, -2.0, 14.5, -2.0, -2.0, 10.5, 3.5,  -2.0, -2.0, 6.5,  -2.0, -2.0,
             5.5,  2.5, 2.5, -2.0, -2.0, 17.5, -2.0, -2.0, 7.5,  4.5,  -2.0, -2.0, 6.5,  -2.0, -2.0},
            {52403230.031,      59106845.39122486,  48367108.57954545, 62614814.93333333,  67925926.16666667, 54648148.083333336,
             40997605.29310345, 43965932.8490566,   9533333.2,         61165879.899782136, 66355820.22619048, 67070707.18181818,
             58492063.71428572, 60003333.266666666, 60213312.30458221, 40527777.5,         44308577.48785872, 25253993.066666666,
             41480555.55,       24199999.6,         47240740.86666667, 19353424.89090909,  29522812.44,       10878935.266666668,
             48089249.0,        43581324.251572326, 41461283.18584071, 48789251.217391305, 51362125.87214612, 52393386.309523806,
             27299382.333333332},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 3, -2, -2, 2, -2, -2, 0, 5, -2, -2, 3, -2, -2, 6, 5, 2, -2, -2, 1, -2, -2, 0, 6, -2, -2, 3, -2, -2},
            {14.5, 5.5, 12.5, 2.5,  -2.0, -2.0, 1.0,  -2.0, -2.0, 11.5, 3.5,  -2.0, -2.0, 2.5,  -2.0, -2.0,
             5.5,  2.5, 4.5,  -2.0, -2.0, 5.5,  -2.0, -2.0, 17.5, 7.5,  -2.0, -2.0, 0.5,  -2.0, -2.0},
            {52838819.419,       61656101.07674944,  52895315.11940298, 59518518.645833336, 53380050.63636363, 64712606.961538464, 36163011.473684214,
             63444444.0,         32953431.17647059,  63217198.57446808, 65389054.35119048,  65950532.84931507, 61662878.86363637,  61463007.370192304,
             60217115.941176474, 63816357.847222224, 45826151.96050269, 22545355.84375,     37473691.15384615, 28092013.7,         43337239.5625,
             17000545.585714284, 18745995.677419353, 3473307.375,       50674213.62472885,  55552647.8495935,  50469617.97115385,  59275430.295774646,
             45092377.255813956, 17788888.6,         45742460.31904762},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 5, -2, -2, 2, -2, -2, 0, 6, -2, -2, 4, -2, -2, 6, 5, 0, -2, -2, 6, -2, -2, 0, 2, -2, -2, 4, -2, -2},
            {15.5, 5.5, 12.5, 2.5,  -2.0, -2.0, 0.5,  -2.0, -2.0, 12.5, 7.5,  -2.0, -2.0, 4.5,  -2.0, -2.0,
             6.5,  1.5, 18.5, -2.0, -2.0, 5.5,  -2.0, -2.0, 17.5, 0.5,  -2.0, -2.0, 2.5,  -2.0, -2.0},
            {52329854.138,       59635002.40175439,  50631465.551724136, 58049107.25,        62540598.41025641,  47745098.11764706,
             37231854.741935484, 63259259.0,         34443204.28571428,  61256757.48654244,  63414351.82478633,  60010912.72448979,
             65866830.0,         59229138.71084338,  60221366.45023697,  53719663.631578945, 42646285.509302326, 30582216.342657343,
             47432950.20689655,  52618055.5,         41051282.15384615,  26295626.149122808, 18377199.033333335, 35093878.5,
             48657302.55052265,  55150735.323529415, 37027777.75,        55890447.87755102,  45077139.61621621,  54887096.67741936,
             43102408.0},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 5, 0, -2, -2, 2, -2, -2, 0, 6, -2, -2, 6, -2, -2, 6, 6, 5, -2, -2, 3, -2, -2, 5, 1, -2, -2, 3, -2, -2},
            {17.5, 5.5, 2.5, 10.5, -2.0, -2.0, 3.5,  -2.0, -2.0, 13.5, 7.5,  -2.0, -2.0, 7.5,  -2.0, -2.0,
             6.5,  5.5, 2.5, -2.0, -2.0, 3.5,  -2.0, -2.0, 2.5,  3.5,  -2.0, -2.0, 3.5,  -2.0, -2.0},
            {51658381.058,      56950270.580601096, 41159799.49504951, 52940767.90196078,  63404762.071428575, 48981418.75675676,  29143211.72,
             35482868.61764706, 15671440.8125,      59477746.93502377, 63060864.74923547,  60481032.30597015,  64852043.751295336, 55623537.97039474,
             51096675.33858268, 58871625.847457625, 37204414.15298507, 24850150.818181816, 20745491.741935484, 32478340.0,         14735984.097560976,
             31728228.18918919, 23293209.833333332, 39719298.21052632, 44441526.99408284,  54429292.90909091,  28583333.5,         56096774.161290325,
             42018024.97058824, 46049572.692307696, 38327171.42253521},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 0, 5, 5, -2, -2, 2, -2, -2, 5, 2, -2, -2, 0, -2, -2, 0, 5, 4, -2, -2, 1, -2, -2, 6, 1, -2, -2, 0, -2, -2},
            {5.5,  12.5, 2.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 2.5, 6.0,  -2.0, -2.0, 13.5, -2.0, -2.0,
             15.5, 6.5,  6.5, -2.0, -2.0, 2.0,  -2.0, -2.0, 6.5,  4.5, -2.0, -2.0, 17.5, -2.0, -2.0},
            {52476561.613,       32263255.951388888, 56710114.12820513,  61957341.46428572,  67461538.61538461,  57187037.266666666,
             43353535.45454545,  67777778.0,         29396825.42857143,  23182994.34285714,  35545608.05405405,  31428472.166666668,
             53190476.14285714,  16456278.05882353,  43636573.833333336, 13825926.85483871,  55876930.78971963,  61309027.7161017,
             61649899.036324784, 62024674.561822124, 36968253.71428572,  21427083.25,        61333333.0,         8125000.0,
             49199978.317708336, 36949010.60273973,  31107905.903846152, 51412698.428571425, 52075607.395498395, 56752567.72268908,
             49176866.359375},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {0, 6, 0, 1, -2, -2, 5, -2, -2, 0, 6, -2, -2, 1, -2, -2, 6, 6, 5, -2, -2, 3, -2, -2, 0, 3, -2, -2, 4, -2, -2},
            {15.5, 5.5, 12.5, 3.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 13.5, 7.5,  -2.0, -2.0, 5.5,  -2.0, -2.0,
             6.5,  5.5, 1.5,  -2.0, -2.0, 1.5,  -2.0, -2.0, 17.5, 0.5,  -2.0, -2.0, 2.5,  -2.0, -2.0},
            {52244045.126,       59740368.393382356, 47330100.62068965, 55281045.88235294,  51367521.43589743,  68000000.33333333,
             36066261.5,         54606060.27272727,  27908750.04,       62102935.78118162,  63908668.56896552,  61718843.83783784,
             65266604.575418994, 58967232.13772455,  59303535.29090909, 31222222.0,         43301062.98245614,  29460589.140127387,
             23306233.085365854, 47641975.44444445,  20305936.08219178, 36189351.76,        62172839.44444445,  32646148.893939395,
             50568468.97993311,  56750778.78504673,  7736111.0,         57684391.504761904, 47123119.244791664, 54238888.9,
             45250548.28289474},
        },
        DecisionTree<32>{
            31,
            {1, 2, 3, 4, -1, -1, 7, -1, -1, 10, 11, -1, -1, 14, -1, -1, 17, 18, 19, -1, -1, 22, -1, -1, 25, 26, -1, -1, 29, -1, -1},
            {16, 9, 6, 5, -1, -1, 8, -1, -1, 13, 12, -1, -1, 15, -1, -1, 24, 21, 20, -1, -1, 23, -1, -1, 28, 27, -1, -1, 30, -1, -1},
            {6, 0, 0, 3, -2, -2, 1, -2, -2, 5, 1, -2, -2, 1, -2, -2, 0, 0, 5, -2, -2, 6, -2, -2, 6, 5, -2, -2, 0, -2, -2},
            {5.5,  14.5, 10.5, 1.5,  -2.0, -2.0, 2.5,  -2.0, -2.0, 1.5, 2.5,  -2.0, -2.0, 5.5,  -2.0, -2.0,
             15.5, 10.5, 3.5,  -2.0, -2.0, 6.5,  -2.0, -2.0, 6.5,  2.5, -2.0, -2.0, 17.5, -2.0, -2.0},
            {52042634.535,       33532227.819277108, 49093309.83098592, 62059027.875,      42763889.0,         68490740.83333333, 45321464.58181818,
             52908119.538461536, 38519636.0,         21902366.52631579, 44194444.25,       62083333.25,        38231481.25,       17387515.34177215,
             19376899.608695652, 3660763.9,          55726960.09232614, 61234956.22538293, 66567688.551724136, 67678062.85897435, 56944444.55555555,
             59981043.48918919,  54717483.65882353,  61550877.12280702, 49050158.4137931,  40520551.7027027,   32086507.85714286, 48089565.41025641,
             51133296.686468646, 56696355.80733945,  48007660.37628866},
        },
    };

    auto x = array<double, 7>{};
    x[0] = common::N;
    rep(i, common::N) x[(int)common::pet_types[i]]++;
    x[6] = common::M;

    const auto predicted_score_long = reg_long.Predict(x);
    const auto predicted_score_short = reg_short.Predict(x);
    cout << "#predicted_score_long=" << predicted_score_long << endl;
    cout << "#predicted_score_short=" << predicted_score_short << endl;
    if (predicted_score_long < predicted_score_short) {
        SHORT_BAR = true;
    } else {
        SHORT_BAR = false;
    }
    cout << "#SHORT_BAR=" << SHORT_BAR << endl;
}

void Solve() {
    Initialize();
    SetBarLength();
    PreComputeFeatures();
    rep(_, 300) {
        MakeAction();
        UpdateHuman();
        Interact();
        UpdatePets();
        PreComputeFeatures();
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
