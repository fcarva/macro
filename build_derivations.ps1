$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

$targets = @(
    (Join-Path $repoRoot "ch01_solow\notes\ch01_solow_derivations.tex"),
    (Join-Path $repoRoot "ch02_rck_diamond\notes\ch02_rck_diamond_derivations.tex"),
    (Join-Path $repoRoot "notes\guia_derivacoes_ch01_ch02.tex")
)

function Invoke-LatexBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TexPath
    )

    $resolved = Resolve-Path $TexPath
    $source = $resolved.Path
    $outputDir = Split-Path $source -Parent

    Write-Host "Compiling $source"
    foreach ($pass in 1..2) {
        Write-Host "  Pass $pass"
        & pdflatex `
            -interaction=nonstopmode `
            -halt-on-error `
            -file-line-error `
            -output-directory $outputDir `
            $source
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex failed for $source on pass $pass"
        }
    }

    Get-ChildItem -LiteralPath $outputDir -File |
        Where-Object { $_.Extension -in @(".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk") } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

foreach ($target in $targets) {
    Invoke-LatexBuild -TexPath $target
}

Write-Host "Derivation notes compiled successfully."
