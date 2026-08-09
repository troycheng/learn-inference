#!/usr/bin/env ruby

require "json"
require "net/http"
require "uri"

files = [
  "README.md",
  "docs/roadmap.md",
  "docs/inference-map.md",
  "docs/glossary.md",
  "docs/capstone.md",
] + Dir["docs/lessons/[0-9][0-9]-*.md"].sort

repository = ENV.fetch("GITHUB_REPOSITORY", "troycheng/learn-inference")
token = ENV["GITHUB_TOKEN"]
endpoint = URI("https://api.github.com/markdown")
errors = []

files.each do |file|
  request = Net::HTTP::Post.new(endpoint)
  request["Accept"] = "application/vnd.github+json"
  request["User-Agent"] = "learn-inference-course-check"
  request["X-GitHub-Api-Version"] = "2022-11-28"
  request["Authorization"] = "Bearer #{token}" unless token.to_s.empty?
  request.body = JSON.generate(
    text: File.read(file),
    mode: "gfm",
    context: repository,
  )

  http = Net::HTTP.new(endpoint.host, endpoint.port)
  http.use_ssl = true
  http.open_timeout = 10
  http.read_timeout = 30
  response = http.start { |connection| connection.request(request) }

  unless response.is_a?(Net::HTTPSuccess)
    errors << "#{file}: GitHub Markdown API returned #{response.code}"
    next
  end

  html = response.body
  errors << "#{file}: GitHub rendered a math error" if html.include?("flash flash-error")
  errors << "#{file}: GitHub rejected a math macro" if html.include?("The following macros are not allowed")
  errors << "#{file}: pasted GitHub error text reached rendered HTML" if html.include?("There was an error in your Markdown")
end

unless errors.empty?
  warn errors.join("\n")
  exit 1
end

puts "GitHub GFM render: PASS (#{files.length} reader-facing files)"
