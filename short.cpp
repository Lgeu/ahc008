#define SKIP_MAIN
#include "answer.cpp"
#undef SKIP_MAIN

int main() {
    SHORT_BAR = true;
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
