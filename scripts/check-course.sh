#!/usr/bin/env bash

set -euo pipefail

ruby <<'RUBY'
errors = []
files = ["README.md"] + Dir["docs/**/*.md"]

files.each do |file|
  text = File.read(file)
  lines = text.lines

  code_fences = lines.count { |line| line.start_with?("```") }
  math_fences = lines.count { |line| line.strip == "$$" }
  details_open = text.scan(/<details(?:\s[^>]*)?>/).length
  details_close = text.scan(%r{</details>}).length

  errors << "#{file}: code fence count is odd" if code_fences.odd?
  errors << "#{file}: display-math fence count is odd" if math_fences.odd?
  errors << "#{file}: <details> tags are not balanced" if details_open != details_close
  errors << "#{file}: GitHub does not allow \\operatorname here" if text.include?("\\operatorname")
  errors << "#{file}: display math appears in a heading" if lines.any? { |line| line.start_with?("#") && line.include?("$$") }

  image_counts = Hash.new(0)
  text.scan(/!\[[^\]]*\]\(([^)]+)\)/).flatten.each do |target|
    image_counts[target.split(/[?#]/).first] += 1
  end
  image_counts.each do |target, count|
    errors << "#{file}: the same image is referenced #{count} times: #{target}" if count > 1
  end

  text.scan(/!?\[[^\]]*\]\(([^)]+)\)/).flatten.each do |raw_target|
    target = raw_target.split(/\s+[\"\047]/, 2).first.to_s.sub(/\A</, "").sub(/>\z/, "")
    next if target.empty? || target.match?(%r{\A(?:https?://|mailto:|#)})

    local_target = target.split(/[?#]/).first
    resolved = File.expand_path(local_target, File.dirname(file))
    errors << "#{file}: missing local link: #{target}" unless File.exist?(resolved)
  end
end

unless errors.empty?
  warn errors.join("\n")
  exit 1
end

puts "Markdown structure and local links: PASS (#{files.length} files)"
RUBY

svg_render_target=$(mktemp "${TMPDIR:-/tmp}/learn-inference-svg.XXXXXX.png")
trap 'unlink "$svg_render_target"' EXIT

svg_count=0
for file in docs/assets/*.svg; do
  xmllint --noout "$file"
  rsvg-convert "$file" -o "$svg_render_target"
  svg_count=$((svg_count + 1))
done
echo "SVG XML and render: PASS (${svg_count} files)"

python3 examples/attention_walkthrough.py > /dev/null
echo "Runnable examples: PASS"

git diff --check
echo "Whitespace check: PASS"
