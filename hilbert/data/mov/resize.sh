#!/bin/bash
find . -name '*.bmp' -execdir mogrify -resize 64x64! {} +
