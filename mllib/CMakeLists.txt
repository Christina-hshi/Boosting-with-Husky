# Copyright 2016 Husky Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

add_library(test-objs OBJECT
    testcompile.cpp)
set_property(TARGET test-objs PROPERTY CXX_STANDARD 14)

# add cpp files in mllib directory in 'mllib-objs'
add_library(mllib-objs OBJECT
	DataReader.cpp
	MaxAbsScaler.cpp
	LinearRegression.cpp
	LinearRegression_SGD.cpp
	SimpleLinearRegression.cpp
	)
set_property(TARGET mllib-objs PROPERTY CXX_STANDARD 14)

# Visible to parent directory
set(mllib-objs $<TARGET_OBJECTS:mllib-objs> PARENT_SCOPE)

#$<TARGET_OBJECTS:mllib-objs> 
add_library(husky-mllib STATIC $<TARGET_OBJECTS:mllib-objs>)
