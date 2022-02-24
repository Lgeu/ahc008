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
            WAITING,
        };
        Type type;
        bool setting_left_to_right;
        Vec2<i8> moving_target_position;
        Vec2<i8> setting_target_position;
    };
    static auto human_states = array<HumanState, 10>();

    rep(idx_human, common::M) {
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
        setting:;
        }
    }

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
        ok:;
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
