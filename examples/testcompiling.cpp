
#include "mllib/testcompile.hpp"
#include "core/engine.hpp"

int main(int argc, char** argv) {
    if (husky::init_with_args(argc, argv)) {
        husky::run_job(pi);
        return 0;
    }
    return 1;
}
