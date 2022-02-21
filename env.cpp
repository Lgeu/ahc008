#include <arpa/inet.h> //バイトオーダの変換に利用
#include <cstdlib>
#include <string.h>
#include <sys/socket.h> //アドレスドメイン
#include <sys/types.h>  //ソケットタイプ
#include <unistd.h>     //close()に利用

#define SKIP_MAIN
#include "answer.cpp"
#undef SKIP_MAIN

namespace connection {

auto port = 1234;
auto sockfd = 0;
struct sockaddr_in addr;

} // namespace connection

void InitConnection() {
    using namespace connection;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    cout << "#port=" << port << endl;
    connect(sockfd, (struct sockaddr*)&addr, sizeof(struct sockaddr_in));
}

template <typename T> void Send(const T& data) {
    using namespace connection;
    auto sum_sent = 0;
    while (sum_sent < (int)sizeof(data)) {
        auto sent = send(sockfd, (char*)&data + sum_sent, sizeof(data) - sum_sent, 0);
        if (sent < 0) {
            exit(1);
        }
        sum_sent += sent;
    }
}

template <typename T> void Recv(T& data) {
    using namespace connection;
    auto sum_recd = 0;
    while (sum_recd < (int)sizeof(data)) {
        auto recd = recv(sockfd, (char*)&data + sum_recd, min((int)sizeof(data) - sum_recd, 2048), 0);
        if (recd < 0) {
            exit(1);
        }
        sum_recd += recd;
    }
}

void MakeEnvironment() {
    InitConnection();
    Initialize();
    Send((i8)common::M);
    PreComputeFeatures();
    ComputeReward();

    rep(_, 300) {
        ExtractFeatures();
        Send(features::observation_global);
        Send(features::observation_local);
        Send(common::human_positions);
        ComputeLegalActions();
        rep(i, common::M) { Send((short)common::legal_actions[i]); }
        rep(i, common::M) {
            auto action = (i8)8;
            Recv(action);
            common::human_moves[i] = "udlrUDLR."[action];
        }
        // Predict();
        UpdateHuman();
        ComputeReward();
        rep(i, common::M) { Send((float)rl::reward[i]); }
        if (common::current_turn == 299)
            break;
        Interact();
        PreComputeFeatures();
        UpdatePets();
        common::current_turn++;
    }
    ComputeOutcome();
    rep(i, common::M) { Send((float)rl::outcome[i]); }
    Interact();
}

int main(int argc, char* argv[]) {
    // TODO
    assert(argc >= 2);
    connection::port = atoi(argv[1]);
    if (argc >= 3) {
        rl::log_reward_ratio = atof(argv[2]);
    }
    if (argc >= 4) {
        rl::reward_coef = atof(argv[3]);
    }
    MakeEnvironment();
}