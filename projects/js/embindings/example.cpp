//
// Created by maksim on 11/14/24.
//

#include <emscripten/bind.h>
#include <string>
#include <vector>
#include <optional>

using namespace emscripten;

std::vector<int> returnVectorData () {
  std::vector<int> v(10, 1);
  return v;
}

std::map<int, std::string> returnMapData () {
  std::map<int, std::string> m;
  m.insert(std::pair<int, std::string>(10, "This is a string."));
  return m;
}

std::optional<std::string> returnOptionalData() {
  return "hello";
}

EMSCRIPTEN_BINDINGS(module) {
  function("returnVectorData", &returnVectorData);
  function("returnMapData", &returnMapData);
  function("returnOptionalData", &returnOptionalData);

  // register bindings for std::vector<int>, std::map<int, std::string>, and
  // std::optional<std::string>.
  register_vector<int>("vector<int>");
  register_map<int, std::string>("map<int, string>");
  register_optional<std::string>();
}
