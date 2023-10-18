#include <iostream>
#include <string>

void run(int sharedMemorySize);

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: shared-memory-limit <shared-memory-size>" << std::endl;
        return 1;
    }

    run(std::stoi(argv[1]));

    return 0;
}
