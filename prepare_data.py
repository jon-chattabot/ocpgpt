from os import walk, sep, system, rename, getenv, path, getcwd
from collections.abc import Callable
from shutil import copyfile
from dotenv import load_dotenv

load_dotenv()

def walk_and_do(doc_source_dir:str, cb:Callable[[str, str], None]) -> None:
    for root, _dirs, files in walk(doc_source_dir):
        for file in files:
            cb(file, root)

def convert_to_html(file:str, root:str) -> None:
    _filename, file_extension = path.splitext(file)
    full_path = f"{root}{sep}{file}"
    if file_extension == ".adoc":
        print(f"processing {full_path}")
        system(f"asciidoctor-pdf {full_path}")

def move_to_docs(file:str, root:str) -> None:
    _filename, file_extension = path.splitext(file)
    full_path = f"{root}{sep}{file}"
    try:
        if file_extension == ".html":
            # print("moving", full_path, f"{getcwd()}{sep}docs{sep}{file}")
            rename(full_path, f"{getcwd()}{sep}docs{sep}{file}")
    except Exception as err:
        print(f"Error in moving file {full_path}", err)

def copy_to_docs(file:str, root:str) -> None:
    _filename, file_extension = path.splitext(file)
    full_path = f"{root}{sep}{file}"
    try:
        if file_extension == ".adoc":
            # print("copying", full_path, f"{getcwd()}{sep}docs{sep}{file}")
            copyfile(full_path, f"{getcwd()}{sep}docs{sep}{file}")
    except Exception as err:
        print(f"Error in copying file {full_path}", err)

def main(doc_source_dir:str):
    # walk_and_do(doc_source_dir, convert_to_html)
    # walk_and_do(doc_source_dir, move_to_docs)
    # walk_and_do(doc_source_dir, copy_to_docs)

if __name__ == '__main__':
    main(getenv("DOCUMENT_SOURCE_DIR", "."))
