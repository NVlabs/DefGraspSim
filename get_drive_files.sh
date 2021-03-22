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

# This script downloads the `drive_files` directory zip file from Google Drive and upzips the
# contents to the proper location.

# NOTE(roflaherty): To update drive_files_v#.#.zip do the following:
#
# * Run the `move_git_lfs_files.sh` script to update the `drive_files`
#   directory with any new Git LFS files.
#
# * Create the zip file with the correct version number (should match the code
#   release version number). Something like:,
#
#   zip -r drive_files_v1.0.zip drive_files
#
# * Upload the zip file to google drive folder.
#
#   Upload to
#   https://drive.google.com/drive/folders/1fxkK8qblET5ec26NotgNUxuXD9_Pj2RL?usp=sharing
#
# * Change the permissions of the file in google drive to "Anyone with the
#   link" by right-clicking on the file and selecting "Get link"
#
# * Update the `FILE_ID` variable below (extract from the public google drive link).
#
#   If the link to the file is
#   https://drive.google.com/file/d/1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD/view?usp=sharing
#   Then file ID is 1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD


# Version 1.1
FILE_ID=1WqELmrv35Mlng2ueaGXvNYH3VqiBTymb

# Version 1.0
# FILE_ID=1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD


# Check if wget and unzip are installed, if not exit
WGET_OK=$(dpkg-query -W -f='${Status}' wget 2>/dev/null | grep -c "ok installed")
UNZIP_OK=$(dpkg-query -W -f='${Status}' unzip 2>/dev/null | grep -c "ok installed")
if [ $WGET_OK -eq 0 ] || [ $UNZIP_OK -eq 0 ]; then
  echo "ERROR: wget and/or unzip are not installed. Install with the following command, then run this script again."
  echo "sudo apt install wget unzip"
  exit 1
fi

# Download zip file from Google Drive
wget --no-check-certificate -O drive_files.zip "https://drive.google.com/uc?export=download&id=$FILE_ID"

# Unzip zip file contents to `./drive_files`
unzip drive_files.zip -d .

# Remove zip file
rm drive_files.zip
