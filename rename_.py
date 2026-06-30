# -*- coding: utf-8 -*-
import os


def batch_rename_files(root_dir, old_str, new_str, dry_run=True):
    """
    批量重命名文件，将文件名中的 old_str 替换为 new_str

    参数:
        root_dir (str): 要搜索的根目录路径
        old_str (str): 文件名中需要被替换的旧字符串
        new_str (str): 用于替换的新字符串
        dry_run (bool): 试运行模式，True=只显示不实际执行，False=实际执行重命名
    """
    renamed_count = 0
    total_affected = 0

    print(f"{'试运行模式' if dry_run else '实际执行模式'}")
    print(f"根目录: {os.path.abspath(root_dir)}")
    print(f"替换规则: '{old_str}' -> '{new_str}'")
    print("-" * 60)

    # 用于收集需要重命名的文件信息
    rename_plan = []

    # 遍历所有文件和子目录[1,4](@ref)
    for current_dir, dirs, files in os.walk(root_dir):
        for filename in files:
            # 检查文件名是否包含要替换的字符串[1](@ref)
            if old_str in filename:
                total_affected += 1

                # 生成新文件名
                new_filename = filename.replace(old_str, new_str)
                old_file_path = os.path.join(current_dir, filename)
                new_file_path = os.path.join(current_dir, new_filename)

                # 检查新文件名是否已存在（且不是同一个文件）[5,6](@ref)
                skip_reason = None
                if os.path.exists(new_file_path) and old_file_path != new_file_path:
                    skip_reason = "目标文件已存在"
                elif old_file_path == new_file_path:
                    skip_reason = "新老文件名相同"

                rename_plan.append({
                    'old_path': old_file_path,
                    'new_path': new_file_path,
                    'old_name': filename,
                    'new_name': new_filename,
                    'skip_reason': skip_reason
                })

    # 显示重命名计划并执行
    for plan in rename_plan:
        if plan['skip_reason']:
            print(f"× 跳过: {plan['old_name']} -> {plan['new_name']} ({plan['skip_reason']})")
        else:
            if dry_run:
                print(f"[试运行] ✓ {plan['old_name']} -> {plan['new_name']}")
            else:
                try:
                    os.rename(plan['old_path'], plan['new_path'])
                    print(f"✓ 已重命名: {plan['old_name']} -> {plan['new_name']}")
                    renamed_count += 1
                except Exception as e:
                    print(f"× 错误重命名 {plan['old_name']}: {e}")

    print("-" * 60)
    if dry_run:
        print(f"试运行完成: 找到 {total_affected} 个需要重命名的文件")
        print(f"实际可重命名: {len([p for p in rename_plan if not p['skip_reason']])} 个文件")
    else:
        print(f"实际操作完成: 成功重命名 {renamed_count}/{total_affected} 个文件")

    return renamed_count, total_affected


def main():
    """主函数：执行批量重命名操作"""
    # ==================== 配置区域 ====================
    ROOT_DIRECTORY = "."  # 要搜索的根目录，"."表示当前目录
    OLD_STRING = "MHM_thld_"  # 要替换的旧字符串
    NEW_STRING = "MHM_thld_050_"  # 替换成的新字符串

    # 安全选项
    DRY_RUN = True  # 试运行模式（True=只显示不实际重命名，False=实际执行）
    REQUEST_CONFIRMATION = True  # 操作前请求确认
    # ==================== 配置结束 ====================

    print("=" * 60)
    print("           文件批量重命名工具")
    print("=" * 60)

    # 首先进行试运行
    success_count, total_files = batch_rename_files(
        ROOT_DIRECTORY, OLD_STRING, NEW_STRING, dry_run=True
    )

    if total_files == 0:
        print("未找到需要重命名的文件。")
        return

    # 询问是否执行实际重命名
    if DRY_RUN and REQUEST_CONFIRMATION:
        print("\n" + "=" * 50)
        confirm = input("是否执行实际重命名操作？(y/N): ").strip().lower()
        if confirm == 'y':
            DRY_RUN = False
            print("开始实际重命名...")
            print("=" * 50)

            # 执行实际重命名
            success_count, total_files = batch_rename_files(
                ROOT_DIRECTORY, OLD_STRING, NEW_STRING, dry_run=False
            )
        else:
            print("操作已取消。")
            return

    print(
        f"\n操作完成！{f'成功重命名 {success_count}/{total_files} 个文件' if not DRY_RUN else '请根据试运行结果确认操作'}")


if __name__ == "__main__":
    main()
