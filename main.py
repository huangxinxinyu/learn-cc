#!/usr/bin/env python3
"""
统一入口 - main.py
使用方法: python main.py [模块名]
如果不带参数，会显示可用的模块列表
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MODULES = {
    "s01": {
        "name": "s01-the-agent-loop",
        "description": "Agent Loop 基础实现（单工具）",
        "module": "s01_the_agent_loop.agent_loop",
        "entry": "main"
    },
    "s02": {
        "name": "s02-tool-use",
        "description": "多工具支持（bash, read, write, edit）",
        "module": "s02.tool_use",
        "entry": "main"
    },
}

def show_help():
    print("用法: python main.py [模块名]")
    print("\n可用的模块:")
    for key, info in MODULES.items():
        print(f"  {key:10s} - {info['description']}")
    print("\n示例:")
    print("  python main.py s01")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    module_key = sys.argv[1]

    if module_key not in MODULES:
        print(f"错误: 未知模块 '{module_key}'")
        show_help()
        sys.exit(1)

    module_info = MODULES[module_key]
    module_path = module_info["module"]

    try:
        from importlib import import_module
        module = import_module(module_path)
        entry_func = getattr(module, module_info["entry"])
        entry_func()
    except ModuleNotFoundError:
        print(f"错误: 模块 '{module_key}' 未找到")
        print(f"提示: 请检查 {module_info['name']} 是否已创建")
        sys.exit(1)
    except AttributeError:
        print(f"错误: 模块 '{module_key}' 没有 '{module_info['entry']}' 函数")
        sys.exit(1)
