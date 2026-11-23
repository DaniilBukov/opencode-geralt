param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = "Stop"

try {
    if (-not (Test-Path -LiteralPath $Path)) {
        [Console]::Error.WriteLine("Audio file not found: $Path")
        exit 1
    }

    if ([IO.Path]::GetExtension($Path) -ieq ".wav") {
        try {
            $sound = New-Object System.Media.SoundPlayer -ArgumentList $Path
            $sound.PlaySync()
            exit 0
        } catch {
            # Fall back to MediaPlayer for WAV variants SoundPlayer cannot play.
        }
    }

    Add-Type -AssemblyName PresentationCore
    $player = New-Object System.Windows.Media.MediaPlayer
    $player.Open([uri]$Path)
    $player.Play()

    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        if ($player.NaturalDuration.HasTimeSpan -and $player.Position -ge $player.NaturalDuration.TimeSpan) {
            break
        }
        Start-Sleep -Milliseconds 100
    }

    $player.Close()
    exit 0
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
