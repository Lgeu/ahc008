#include <arpa/inet.h> //バイトオーダの変換に利用
#include <string.h>
#include <sys/socket.h> //アドレスドメイン
#include <sys/types.h>  //ソケットタイプ
#include <unistd.h>     //close()に利用

#define SKIP_MAIN
#include "answer.cpp"
#undef SKIP_MAIN

namespace connection {

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
    addr.sin_port = htons(1234);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    connect(sockfd, (struct sockaddr*)&addr, sizeof(struct sockaddr_in));
}

template <typename T> void Send(const T& data) {
    using namespace connection;
    auto sum_sent = 0;
    while (sum_sent < sizeof(data)) {
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
    while (sum_recd < sizeof(data)) {
        auto recd = recv(sockfd, (char*)&data + sum_recd, min((int)sizeof(data) - sum_recd, 2048), 0);
        if (recd < 0) {
            exit(1);
        }
        sum_recd += recd;
    }
}

void MakeEnvironment() {
    Initialize();
    Send((i8)common::M);
    PreComputeFeatures();
    rep(_, 300) {
        ExtractFeatures();
        Send(features::observation_global);
        Send(features::observation_local);
        ComputeLegalActions();
        rep(i, common::M) {
            Send((short)common::legal_actions[i]); // TODO
        }
        rep(i, common::M) {
            auto action = (i8)8;
            Recv(action);
            common::human_moves[i] = "udlrUDLR."[action];
        }
        // Predict();
        UpdateHuman();
        ComputeReward();
        rep(i, common::M) {
            Send((float)reward[i]); // TODO
        }
        if (common::current_turn == 299)
            break;
        Interact();
        PreComputeFeatures();
        UpdatePets();
        common::current_turn++;
    }
    ComputeOutCome();
    rep(i, common::M) {
        Send((float)outcome[i]); // TODO
    }
    Interact();
}

int main(int argc, char* argv[]) {
    // TODO
    MakeEnvironment();
}