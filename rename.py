import fnmatch
import os
import re


def find_folders(root_directory, target_pattern, match_type="exact", search_subfolders=True):
    """
    在指定根目录下递归查找匹配特定模式的文件夹

    参数:
        root_directory (str): 要搜索的根目录路径
        target_pattern (str or list): 要查找的文件夹名模式，可以是字符串或列表
        match_type (str): 匹配类型，"exact"-精确匹配, "contains"-包含, "wildcard"-通配符, "regex"-正则表达式
        search_subfolders (bool): 是否搜索子文件夹

    返回:
        list: 找到的文件夹路径列表
    """
    found_folders = []

    if not os.path.exists(root_directory):
        print(f"错误：根目录 '{root_directory}' 不存在")
        return found_folders

    # 如果target_pattern是字符串，转换为列表
    if isinstance(target_pattern, str):
        target_patterns = [target_pattern]
    else:
        target_patterns = target_pattern

    try:
        if search_subfolders:
            # 递归搜索所有子目录
            for current_dir, dirs, files in os.walk(root_directory):
                for dir_name in dirs:
                    matched = False
                    matched_pattern = None

                    for pattern in target_patterns:
                        if match_type == "exact" and dir_name == pattern:
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "contains" and pattern in dir_name:
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "wildcard" and fnmatch.fnmatch(dir_name, pattern):
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "regex" and re.search(pattern, dir_name):
                            matched = True
                            matched_pattern = pattern

                        if matched:
                            folder_path = os.path.join(current_dir, dir_name)
                            found_folders.append({
                                'path': folder_path,
                                'name': dir_name,
                                'pattern': matched_pattern,
                                'parent': current_dir
                            })
                            break
        else:
            # 只搜索当前目录
            for dir_name in os.listdir(root_directory):
                dir_path = os.path.join(root_directory, dir_name)
                if os.path.isdir(dir_path):
                    matched = False
                    matched_pattern = None

                    for pattern in target_patterns:
                        if match_type == "exact" and dir_name == pattern:
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "contains" and pattern in dir_name:
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "wildcard" and fnmatch.fnmatch(dir_name, pattern):
                            matched = True
                            matched_pattern = pattern
                        elif match_type == "regex" and re.search(pattern, dir_name):
                            matched = True
                            matched_pattern = pattern

                        if matched:
                            found_folders.append({
                                'path': dir_path,
                                'name': dir_name,
                                'pattern': matched_pattern,
                                'parent': root_directory
                            })
                            break

        return found_folders

    except Exception as e:
        print(f"搜索文件夹过程中发生错误: {e}")
        return found_folders


def rename_folders(found_folders, new_name_pattern, rename_type="exact", dry_run=True):
    """
    重命名找到的文件夹[6,7,8](@ref)

    参数:
        found_folders (list): 要重命名的文件夹信息列表
        new_name_pattern (str): 新文件夹名的模式
        rename_type (str): 重命名类型，"exact"-完全替换, "prefix"-添加前缀, "suffix"-添加后缀, "replace"-替换, "regex"-正则替换
        dry_run (bool): 试运行模式，只显示不实际执行

    返回:
        tuple: (成功重命名的文件夹数, 总文件夹数)
    """
    success_count = 0
    total_folders = len(found_folders)

    if total_folders == 0:
        print("没有找到要重命名的文件夹")
        return 0, 0

    print(f"{'试运行模式' if dry_run else '实际执行模式'}: 将处理 {total_folders} 个文件夹")
    print("-" * 60)

    for i, folder_info in enumerate(found_folders, 1):
        old_folder_path = folder_info['path']
        old_folder_name = folder_info['name']
        directory = folder_info['parent']

        try:
            # 根据重命名类型生成新文件夹名[6,8](@ref)
            if rename_type == "exact":
                # 完全替换：直接使用新名称
                new_folder_name = new_name_pattern
            elif rename_type == "prefix":
                # 添加前缀
                new_folder_name = new_name_pattern + old_folder_name
            elif rename_type == "suffix":
                # 添加后缀
                name_part = old_folder_name
                # 如果有扩展名则处理扩展名
                if '.' in old_folder_name:
                    name_part, ext_part = os.path.splitext(old_folder_name)
                    new_folder_name = name_part + new_name_pattern + ext_part
                else:
                    new_folder_name = old_folder_name + new_name_pattern
            elif rename_type == "replace":
                # 文本替换：支持 old->new 格式或简单替换
                if "->" in new_name_pattern:
                    old_part, new_part = new_name_pattern.split("->", 1)
                    new_folder_name = old_folder_name.replace(old_part, new_part)
                else:
                    # 如果不包含->，则完全替换
                    new_folder_name = new_name_pattern
            elif rename_type == "regex":
                # 正则表达式替换
                if "->" in new_name_pattern:
                    pattern, replacement = new_name_pattern.split("->", 1)
                    new_folder_name = re.sub(pattern, replacement, old_folder_name)
                else:
                    new_folder_name = old_folder_name
            else:
                # 默认使用完全替换
                new_folder_name = new_name_pattern

            new_folder_path = os.path.join(directory, new_folder_name)

            # 检查新文件夹是否已存在[7,8](@ref)
            if os.path.exists(new_folder_path) and old_folder_path != new_folder_path:
                print(f"! 警告: 目标文件夹已存在，跳过: {new_folder_name}")
                continue

            if dry_run:
                print(f"[试运行] {i}/{total_folders}: {old_folder_name} -> {new_folder_name}")
            else:
                # 执行实际重命名[7,8](@ref)
                os.rename(old_folder_path, new_folder_path)
                print(f"✓ 已重命名 {i}/{total_folders}: {old_folder_name} -> {new_folder_name}")
                success_count += 1

        except Exception as e:
            print(f"× 错误重命名文件夹 {old_folder_name}: {e}")

    print("-" * 60)
    action = "模拟重命名" if dry_run else "成功重命名"
    print(f"{action}: {success_count if not dry_run else total_folders}/{total_folders} 个文件夹")

    return success_count, total_folders


def display_found_folders(found_folders):
    """显示找到的文件夹信息"""
    if not found_folders:
        print("未找到匹配的文件夹。")
        return

    print(f"找到 {len(found_folders)} 个匹配的文件夹:")
    for i, folder_info in enumerate(found_folders, 1):
        print(f"  {i}. {folder_info['path']} (匹配模式: '{folder_info['pattern']}')")


def main():
    """主函数：文件夹搜索与重命名工具"""
    # ==================== 配置区域 ====================
    ROOT_DIR = "."  # 要搜索的根目录，"."表示当前目录

    # 是否启用文件夹重命名功能
    ENABLE_FOLDER_RENAME = True

    # 文件夹查找配置
    FOLDER_PATTERNS = ["MHM_thld_"]  # 要查找的文件夹名模式
    FOLDER_MATCH_TYPE = "contains"  # 匹配类型: "exact", "contains", "wildcard", "regex"
    SEARCH_SUBFOLDERS = True  # 是否搜索子文件夹

    # 文件夹重命名配置
    FOLDER_NEW_NAME_PATTERN = "MHM_thld_050_"  # 新文件夹名模式
    FOLDER_RENAME_TYPE = "exact"  # 重命名类型: "exact", "prefix", "suffix", "replace", "regex"

    # 安全选项
    DRY_RUN_MODE = True  # 试运行模式（True=只显示不删除，False=实际执行）
    REQUEST_CONFIRMATION = True  # 操作前请求确认
    # ==================== 配置结束 ====================

    print("=" * 70)
    print("           文件夹搜索与重命名工具")
    print("=" * 70)

    # 查找文件夹
    found_folders = find_folders(ROOT_DIR, FOLDER_PATTERNS, FOLDER_MATCH_TYPE, SEARCH_SUBFOLDERS)
    display_found_folders(found_folders)

    if not found_folders:
        print("程序结束。")
        return

    # 试运行模式预览
    if DRY_RUN_MODE:
        print(f"\n{'=' * 60}")
        print("           试运行模式（不会实际重命名）")
        print('=' * 60)

        rename_folders(found_folders, FOLDER_NEW_NAME_PATTERN, FOLDER_RENAME_TYPE, dry_run=True)

        # 询问是否执行实际重命名
        if len(found_folders) > 0:
            confirm = input("\n是否执行实际重命名操作？(y/N): ").strip().lower()
            if confirm == 'y':
                DRY_RUN_MODE = False
                print("切换到实际重命名模式...")
            else:
                print("取消重命名操作。")
                return

    # 实际重命名操作确认
    if not DRY_RUN_MODE and REQUEST_CONFIRMATION:
        print(f"\n⚠️  警告：即将重命名 {len(found_folders)} 个文件夹！")
        print("重命名规则:")
        print(f"  类型: {FOLDER_RENAME_TYPE}")
        print(f"  模式: {FOLDER_NEW_NAME_PATTERN}")
        confirmation = input("请输入 'RENAME' 确认执行重命名操作: ")
        if confirmation != 'RENAME':
            print("操作已取消。")
            return

    # 执行实际重命名
    if not DRY_RUN_MODE:
        print(f"\n{'=' * 60}")
        print("           开始实际重命名操作")
        print('=' * 60)

        success_count, total_folders = rename_folders(
            found_folders, FOLDER_NEW_NAME_PATTERN, FOLDER_RENAME_TYPE, dry_run=False
        )

        print(f"\n操作完成！成功重命名 {success_count}/{total_folders} 个文件夹")


# 提供单独的函数调用接口
def batch_rename_folders(root_dir, patterns, new_pattern,
                         match_type="exact", rename_type="exact",
                         search_subfolders=True, dry_run=True):
    """
    批量重命名文件夹的便捷函数

    参数:
        root_dir (str): 根目录路径
        patterns (list): 要查找的文件夹模式列表
        new_pattern (str): 新名称模式
        match_type (str): 匹配类型
        rename_type (str): 重命名类型
        search_subfolders (bool): 是否搜索子文件夹
        dry_run (bool): 试运行模式

    返回:
        tuple: (成功数, 总数)
    """
    found_folders = find_folders(root_dir, patterns, match_type, search_subfolders)

    if not found_folders:
        print("未找到匹配的文件夹")
        return 0, 0

    display_found_folders(found_folders)
    return rename_folders(found_folders, new_pattern, rename_type, dry_run)


if __name__ == "__main__":
    main()
