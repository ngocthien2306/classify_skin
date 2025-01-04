import os

def find_py_files(root_dir):
   py_files = []
   for root, dirs, files in os.walk(root_dir):
       for file in files:
           if file.endswith('.py'):
               py_files.append(os.path.join(root, file))
   return py_files

def combine_py_contents(files, output='combined_py_files.txt'):
   with open(output, 'w', encoding='utf-8') as out:
       for file in files:
           out.write(f'\n{"-"*80}\n')
           out.write(f'File: {file}\n')
           out.write(f'{"-"*80}\n')
           
           try:
               with open(file, 'r', encoding='utf-8') as f:
                   out.write(f.read())
           except Exception as e:
               out.write(f'Error reading file: {str(e)}\n')


py_files = find_py_files('.')
print(py_files)
combine_py_contents(py_files, 'classify_skin.txt')