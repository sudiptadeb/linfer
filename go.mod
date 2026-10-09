module github.com/sudiptadeb/linfer

go 1.22

require (
	github.com/sudiptadeb/linfer-bench v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

// TODO: once linfer-bench is published and tagged v0.1.0, drop this replace
// and `go get github.com/sudiptadeb/linfer-bench@v0.1.0`.
replace github.com/sudiptadeb/linfer-bench => ../linfer-bench
