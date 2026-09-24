# Logs CPU power sensors (Intel RAPL via LibreHardwareMonitorLib + PawnIO) to CSV at ~10 Hz.
# Must run elevated. Columns: unix_ms, then one column per CPU power sensor (W).
# Stop by creating the file D:\papers-tools\cpu_logger.stop
param(
  [string]$Lib = "D:\papers-tools\LHM-0.9.6\LibreHardwareMonitorLib.dll",
  [string]$Out = "D:\papers-tools\cpu_power.csv",
  [int]$IntervalMs = 100
)
$stop = "D:\papers-tools\cpu_logger.stop"
if (Test-Path $stop) { Remove-Item $stop -Force }
Add-Type -Path $Lib
$c = New-Object LibreHardwareMonitor.Hardware.Computer
$c.IsCpuEnabled = $true
$c.Open()
$cpu = $c.Hardware | Where-Object { $_.HardwareType -eq [LibreHardwareMonitor.Hardware.HardwareType]::Cpu } | Select-Object -First 1
$cpu.Update()
$sensors = @($cpu.Sensors | Where-Object { $_.SensorType -eq [LibreHardwareMonitor.Hardware.SensorType]::Power })
if ($sensors.Count -eq 0) { "NO_POWER_SENSORS" | Out-File -Encoding utf8 $Out; $c.Close(); exit 1 }
$header = "unix_ms," + (($sensors | ForEach-Object { ($_.Name -replace '[ ,]', '_') }) -join ",")
$w = New-Object System.IO.StreamWriter($Out, $false, (New-Object System.Text.UTF8Encoding $false))
$w.AutoFlush = $true
$w.WriteLine($header)
while (-not (Test-Path $stop)) {
  $cpu.Update()
  $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  $vals = $sensors | ForEach-Object { if ($_.Value -ne $null) { [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0:F3}", $_.Value) } else { "" } }
  $w.WriteLine("$t," + ($vals -join ","))
  Start-Sleep -Milliseconds $IntervalMs
}
$w.Close()
$c.Close()
