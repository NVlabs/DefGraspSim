# Copyright (c) 2020 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# NOTE: It is assumed that this script will run from the root of the project.

# This script will create synlinks between files in the `drive_files` directory and the
# corresponding files in the project.

FILE_LIST=drive_files_list.txt
DRIVE_FILES_DIR=drive_files

# Get the current list of all the files in the `drive_files` directory
find $DRIVE_FILES_DIR -type f > $FILE_LIST

# Loop over each file and create a symlink in the proper location within the project directory
# structure.
for a_file in $(cat $FILE_LIST); do
  dest_file=${a_file#$DRIVE_FILES_DIR/}
  touch $dest_file
  realtive_path=$(realpath --relative-to=$(dirname $dest_file) $a_file)
  rm $dest_file
  echo "Running: ln -s $realtive_path $dest_file"
  ln -s $realtive_path $dest_file
done

# Remove FILE_LIST file
rm $FILE_LIST
