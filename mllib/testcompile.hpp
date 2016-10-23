
#include <random>
#include <string>
#include <vector>



class PIObject {
   public:
    typedef int KeyT;
    int key;

    explicit PIObject(KeyT key);

    const int& id() const;
};

void pi();
