param(
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$TextFile = "",
    [string]$Voice = "",
    [int]$Rate = 0
)

$ErrorActionPreference = "Stop"

try {
    Add-Type -AssemblyName System.Speech

    if ($TextFile) {
        $text = Get-Content -LiteralPath $TextFile -Raw -Encoding UTF8
    } else {
        $text = [Console]::In.ReadToEnd()
    }

    if ([string]::IsNullOrWhiteSpace($text)) { exit 0 }

    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        if ($Voice) {
            $synth.SelectVoice($Voice)
        } else {
            $russian = $synth.GetInstalledVoices() |
                Where-Object { $_.VoiceInfo.Culture.Name -like "ru-*" } |
                Select-Object -First 1
            if ($russian) { $synth.SelectVoice($russian.VoiceInfo.Name) }
        }
        $synth.Rate = [Math]::Max(-10, [Math]::Min(10, $Rate))
        $synth.SetOutputToWaveFile($Out)
        $synth.Speak($text)
    } finally {
        $synth.Dispose()
    }
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
