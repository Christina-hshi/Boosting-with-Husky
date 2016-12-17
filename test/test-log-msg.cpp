// Copyright 2016 Husky Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "base/log.hpp"

int main(int argc, char** argv) {
    husky::base::log_info("Info Log");
    husky::base::log_debug("Debug Log");
    husky::base::log_error("Error Log");
    husky::base::log_warning("Warning Log");
    // Test for glog
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG info test";
    LOG(WARNING) << "GLOG warning test";
    LOG(ERROR) << "GLOG error test";
    VLOG(0) << "GLOG vlog 0";
    VLOG(1) << "GLOG vlog 1";
    VLOG(2) << "GLOG vlog 2";
    VLOG(5) << "GLOG vlog 5";
    return 0;
}
