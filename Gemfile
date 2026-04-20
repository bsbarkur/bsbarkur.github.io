source "https://rubygems.org"

# Matches what GitHub Pages builds with, so local == live.
# https://pages.github.com/versions/
gem "github-pages", group: :jekyll_plugins

gem "webrick", "~> 1.7"  # Ruby 3.x removed webrick from stdlib; harmless on 2.6
gem "ffi", "< 1.17"      # ffi 1.17 requires Ruby 3; pin for Ruby 2.6 compat
gem "google-protobuf", "< 3.22"  # newer versions also require Ruby >= 2.7
