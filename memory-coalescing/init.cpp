#include <iostream>

void run(int startOffset, int moveOffset);

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: memory-coalescing <start-offset> <move-offset>" << std::endl;
        return 1;
    }

    run(std::stoi(argv[1]), std::stoi(argv[2]));

    return 0;
}
