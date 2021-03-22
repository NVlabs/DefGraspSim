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

# This script creates a list of all the files that are currently being tracked
# by Git LFS. It moves each of those files to a matching directory in the
# `drive_files` directory and creates a symlink from the original location to
# this new location. It then creates a zip file of the entire `drive_files`
# directory, which then can be upload to Google Drive and later accessed by the
# end user.

GIT_LFS_FILES_TXT=git_lfs_files.txt # Temp file to store the list of Git LFS files
DEST_FOLDER=drive_files # Folder where files will be moved to
ZIP_FILE=$DEST_FOLDER.zip

# Get a list of the current Git LFS files and store them in a text file
git lfs ls-files | awk '{ print $3 }' > $GIT_LFS_FILES_TXT

# Loop over each Git LFS file, moving each file to the `drive_files` folder and
# creating a symlink from the original location to the new location
for a_file in $(cat $GIT_LFS_FILES_TXT); do
  mkdir -p $(dirname $DEST_FOLDER/$a_file)
  cp $a_file $DEST_FOLDER/$a_file
  a_file_real_path=$(realpath --relative-to=$(dirname $a_file) drive_files/$a_file)
  rm $a_file
  echo "Running: ln -s $a_file_real_path $a_file"
  ln -s $a_file_real_path $a_file
done

# Create a zip file of the files in the `drive_files` folder
zip -r $ZIP_FILE $DEST_FOLDER

# Remove `git_lfs_files.txt` file
rm $GIT_LFS_FILES_TXT

# Print info
echo "=================================================="
echo "All Git LFS files were moved into the $DEST_FOLDER "
echo "folder and symlinks were created in the original "
echo "locations to these new files."
echo ""
echo "Additionally, a zip file was created for the "
echo "$DEST_FOLDER folder, called $ZIP_FILE."
echo ""

# Print next steps to the terminal
echo "=================================================="
echo "NEXT STEPS:"
echo ""
echo "Upload this zip file to Google drive folder at this location:"
echo "https://drive.google.com/drive/u/1/folders/1fxkK8qblET5ec26NotgNUxuXD9_Pj2RL"
echo ""
echo "Get the file ID"
echo "* Copy the link to the zip file that was just uploaded"
echo "* Extract out the file ID from the link"
echo "  For example:"
echo "    https://drive.google.com/file/d/1sjcQj7KO0Vp8eD5yKdmM5HaSuhwyP98c/view?usp=sharing"
echo "  The file ID is 1sjcQj7KO0Vp8eD5yKdmM5HaSuhwyP98c"
echo "* Paste the file ID into the get_drive_files.sh file."
