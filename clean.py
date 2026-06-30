import os

def delete_files_by_extension(extension):
    """
    递归删除当前工作目录下所有指定扩展名的文件
    :param extension: 文件扩展名，如 '.csv' 或 'csv'
    """
    # 统一格式：确保扩展名以点开头
    if not extension.startswith('.'):
        extension = '.' + extension

    deleted_count = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除失败: {file_path} - {e}")

    print(f"\n共删除 {deleted_count} 个 {extension} 文件")

if __name__ == "__main__":
    target_ext = input("请输入要删除的文件扩展名（如 csv 或 .csv）：").strip()
    confirm = input(f"即将删除当前目录及其子目录下所有 .{target_ext.lstrip('.')} 文件，是否继续？(y/n): ")
    if confirm.lower() == 'y':
        delete_files_by_extension(target_ext)
    else:
        print("操作已取消")

