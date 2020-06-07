/*
 * testMultipleStreamsMultipleTerminals: test having different streams do i/o in different terminals
 *
 * Evan Herbst
 * 12 / 5 / 13
 */

#include <cstdint>
#include <iostream>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>
#include "rgbd_util/parallelism.h"
using std::endl;
namespace io = boost::iostreams;

int main()
{
#if 1 //using multiple terminals -- if I use an existing terminal, things get written to it as commands; try opening one? maybe with openpty?
	const char* path1 = "/dev/pts/5", *path2 = "/dev/pts/6";
	FILE* file1 = fopen(path1, "a"), *file2 = fopen(path2, "a");
	int fd1 = fileno(file1), fd2 = fileno(file2);
	io::file_descriptor fdstream1(fd1, io::close_handle), fdstream2(fd2, io::close_handle);
	io::stream<io::file_descriptor> stream1(fdstream1), stream2(fdstream2);
	rgbd::thread usuallyNoInputThread([&]()
		{
			const auto func = [](std::istream& in, std::ostream& out)
				{
					uint64_t iter = 0;
					while(true)
					{
						out << "thread 1 on iter " << iter << endl;
						if(rand() % 100 == 0)
						{
							out << "t1 enter number: " << endl;
							int q; in >> q;
						}
						iter++;
					}
				};
			func(stream1, stream1);
		});
	rgbd::thread lotsOfInputThread([&]()
		{
			const auto func = [](std::istream& in, std::ostream& out)
				{
					uint64_t iter = 0;
					while(true)
					{
						out << "thread 2 on iter " << iter << endl;
						out << "t2 enter number: " << endl;
						int q; in >> q;
						iter++;
					}
				};
			func(stream2, stream2);
		});
	usuallyNoInputThread.join();
	lotsOfInputThread.join();
#else //using the same terminal
	rgbd::thread usuallyNoInputThread([&]()
		{
			const auto func = [](std::istream& in, std::ostream& out)
				{
					uint64_t iter = 0;
					while(true)
					{
						out << "thread 1 on iter " << iter << endl;
						if(rand() % 100 == 0)
						{
							out << "t1 enter number: " << endl;
							int q; in >> q;
						}
						iter++;
					}
				};
			func(std::cin, std::cout);
		});
	rgbd::thread lotsOfInputThread([&]()
		{
			const auto func = [](std::istream& in, std::ostream& out)
				{
					uint64_t iter = 0;
					while(true)
					{
						out << "thread 2 on iter " << iter << endl;
						out << "t2 enter number: " << endl;
						int q; in >> q;
						iter++;
					}
				};
			func(std::cin, std::cout);
		});
	usuallyNoInputThread.join();
	lotsOfInputThread.join();
#endif
	return 0;
}
