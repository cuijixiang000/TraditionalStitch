#include "spdloghelper.h"
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/basic_file_sink.h>

SimLog2* SimLog2::Instance()
{
	static SimLog2* log = new SimLog2;
	return log;
}

void SimLog2::InitSimLog(std::string logger_name, std::string file_name, int log_level)
{
	//设置日志等级
	spdlog::set_level(static_cast<spdlog::level::level_enum>(log_level));
	//设置日志为异步日志，不带滚动，日志文件会一直写入
	// my_logger_ = spdlog::basic_logger_mt(logger_name, file_name);

	// 基于滚动文件的日志，每个文件5MB，三个文件
	// my_logger_ = spdlog::rotating_logger_mt(logger_name, file_name, 1024 * 1024 * 5, 3);

	//my_logger_ = spdlog::default_logger();  //1.x 版本
	my_logger_ = spdlog::daily_logger_mt(logger_name, file_name, 0, 0/*, true*/);

	// my_logger_ = spdlog::stdout_logger_mt(logger_name);

	// auto sharedFileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("test.log");
	// my_logger_ = std::make_shared<spdlog::logger>(logger_name, sharedFileSink);
	//当遇到错误级别以上的立刻刷新到日志
	my_logger_->flush_on(spdlog::level::info);
	//每三秒刷新一次
	//spdlog::flush_every(std::chrono::seconds(3));

	//测试
//    for (int i = 0; i < 101; i++)
//    {
//        my_logger_->info("SimLog::Async message #{}", i);
//    }
}

void SimLog2::EndLog()
{
}

SimLog2::SimLog2()
{
	//    InitSimLog("vi_service","test.log");
}

SimLog2::~SimLog2()
{
	EndLog();
}

void SimLog2::SetLevel(int level)
{
	spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}