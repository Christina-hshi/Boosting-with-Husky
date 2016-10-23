
#include "mllib/testcompile.hpp"
int main(int argc, char** argv) {
    if (husky::init_with_args(argc, argv)) {
        husky::run_job(pi);
        return 0;
    }
    return 1;
}
