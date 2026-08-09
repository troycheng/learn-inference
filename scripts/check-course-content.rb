#!/usr/bin/env ruby

errors = []
lesson_files = Dir["docs/lessons/[0-9][0-9]-*.md"].sort
expected_numbers = (0..9).to_a
actual_numbers = lesson_files.map { |file| File.basename(file, ".md")[0, 2].to_i }

if actual_numbers != expected_numbers
  errors << "lesson files must cover 00 through 09 exactly; found #{actual_numbers.inspect}"
end

lesson_titles = {}
referenced_assets = Hash.new { |hash, key| hash[key] = [] }
roadmap = File.read("docs/roadmap.md")

(["README.md"] + Dir["docs/**/*.md"]).each do |file|
  text = File.read(file)
  errors << "#{file}: use L_full for the Full Attention layer count" if text.match?(/\bLfull\b/)
end

lesson_files.each_with_index do |file, lesson_index|
  text = File.read(file)
  lines = text.lines
  h1_lines = lines.select { |line| line.start_with?("# ") }

  if h1_lines.length != 1
    errors << "#{file}: expected one H1, found #{h1_lines.length}"
    next
  end

  title = h1_lines.first.strip
  match = title.match(/\A# 第 (\d+) 课：(.+)\z/)
  if match.nil?
    errors << "#{file}: H1 must use '# 第 N 课：标题'"
  else
    file_number = File.basename(file, ".md")[0, 2].to_i
    title_number = match[1].to_i
    errors << "#{file}: filename and H1 lesson numbers differ" if file_number != title_number
    lesson_titles[file] = title.delete_prefix("# ")

    roadmap_entry = "| [#{title_number}](lessons/#{File.basename(file)}) | #{match[2]} |"
    errors << "docs/roadmap.md: lesson title does not match #{file}" unless roadmap.include?(roadmap_entry)
  end

  numbered_sections = lines.map do |line|
    section_match = line.match(/\A## (\d+)\. /)
    section_match && section_match[1].to_i
  end.compact
  expected_sections = (1..numbered_sections.length).to_a
  if numbered_sections != expected_sections
    errors << "#{file}: numbered H2 sections must be consecutive; found #{numbered_sections.inspect}"
  end

  errors << "#{file}: lesson pages use SVG instead of Mermaid" if text.include?("```mermaid")
  errors << "#{file}: contains an unsupported GitHub math macro" if text.include?("\\operatorname")
  errors << "#{file}: contains a pasted GitHub math error" if text.include?("The following macros are not allowed")

  expected_navigation = ["../roadmap.md"]
  expected_navigation << File.basename(lesson_files[lesson_index - 1]) if lesson_index.positive?
  expected_navigation << File.basename(lesson_files[lesson_index + 1]) if lesson_index < lesson_files.length - 1
  expected_navigation.each do |target|
    errors << "#{file}: missing course navigation link to #{target}" unless text.include?("](#{target})")
  end

  exercise_index = lines.index { |line| line.match?(/\A## \d+\. 练习\s*\z/) }
  details_index = lines.each_index.find do |index|
    exercise_index && index > exercise_index && lines[index].start_with?("<details>")
  end
  details_close_index = lines.each_index.find do |index|
    details_index && index > details_index && lines[index].start_with?("</details>")
  end

  if exercise_index.nil? || details_index.nil? || details_close_index.nil?
    errors << "#{file}: expected a numbered exercise section with folded answers"
  else
    questions = lines[(exercise_index + 1)...details_index].map do |line|
      line[/\A(\d+)\. /, 1]&.to_i
    end.compact
    answers = lines[(details_index + 1)...details_close_index].map do |line|
      line[/\A(\d+)\. /, 1]&.to_i
    end.compact

    expected_questions = (1..questions.length).to_a
    errors << "#{file}: exercise numbers must be consecutive; found #{questions.inspect}" if questions != expected_questions
    errors << "#{file}: exercise answers must match questions; found #{answers.inspect}" if answers != questions
  end

  lines.each_with_index do |line, index|
    next unless line.include?("$$")
    errors << "#{file}:#{index + 1}: display-math delimiters must occupy their own line" unless line.strip == "$$"
  end

  text.scan(/!\[([^\]]*)\]\(([^)]+)\)/).each do |alt, raw_target|
    errors << "#{file}: image alt text must not be empty" if alt.strip.empty?
    target = raw_target.split(/[?#]/).first
    next unless target.end_with?(".svg")

    resolved = File.expand_path(target, File.dirname(file))
    referenced_assets[resolved] << file
  end
end

readme = File.read("README.md")
readme_lessons = readme.scan(/\[(第 \d+ 课：[^\]]+)\]\((docs\/lessons\/[^)]+)\)/)

if readme_lessons.length != lesson_files.length
  errors << "README.md: expected #{lesson_files.length} lesson links, found #{readme_lessons.length}"
end

readme_lessons.each do |label, target|
  expected_title = lesson_titles[target]
  errors << "README.md: lesson label does not match #{target}: #{label.inspect}" if expected_title != label
end

referenced_assets.each do |asset, owners|
  next if owners.length == 1
  errors << "course image is referenced more than once: #{asset} from #{owners.join(', ')}"
end

asset_files = Dir[File.expand_path("docs/assets/*.svg")].sort
orphan_assets = asset_files - referenced_assets.keys
orphan_assets.each { |asset| errors << "unreferenced course image: #{asset}" }

unless errors.empty?
  warn errors.join("\n")
  exit 1
end

puts "Course consistency: PASS (#{lesson_files.length} lessons, #{asset_files.length} unique images)"
