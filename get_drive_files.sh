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

# NOTE(roflaherty): To update drive_files.zip do the following:
#
# * Run the following command,
#
#   zip -r drive_files.zip drive_files
#
# * Upload the zip file to google drive,
#
#   Upload to https://drive.google.com/drive/folders/1fxkK8qblET5ec26NotgNUxuXD9_Pj2RL?usp=sharing
#
# * Change the name of file to have the correct version number (should match code release version number).
#
# * Change the permissions of the file to "Anyone with the link" by right-clicking on the file and
#   selecting "Get link"
#
# * Update the `FILE_ID` variable below (extract from the public google drive link).
#
#   If the link to the file is https://drive.google.com/file/d/1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD/view?usp=sharing
#   Then file ID is 1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD

FILE_ID=1pFv9rsaj-eZp9VQt654i0HgfdXv3-LuD

# Install wget and unzip
sudo apt install wget unzip

# Download zip file from Google Drive
wget --no-check-certificate -O drive_files.zip "https://drive.google.com/uc?export=download&id=$FILE_ID"

# Unzip zip file contents to `./drive_files`
unzip drive_files.zip -d .
