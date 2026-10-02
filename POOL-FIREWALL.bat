@echo off
rem Strata's pool: let the pool's engines, beacons and web app through Windows Firewall on PRIVATE networks.
rem Run it once on every PC of the pool (right-click, "Run as administrator").
net session >nul 2>&1
if errorlevel 1 (
  echo Right-click POOL-FIREWALL.bat and choose "Run as administrator".
  pause
  exit /b 1
)
netsh advfirewall firewall delete rule name="Strata pool engine (TCP 7701)" >nul 2>&1
netsh advfirewall firewall delete rule name="Strata pool discovery (UDP 7702)" >nul 2>&1
netsh advfirewall firewall delete rule name="Strata pool web app (TCP 8080)" >nul 2>&1
netsh advfirewall firewall add rule name="Strata pool engine (TCP 7701)" dir=in action=allow protocol=TCP localport=7701 profile=private
netsh advfirewall firewall add rule name="Strata pool discovery (UDP 7702)" dir=in action=allow protocol=UDP localport=7702 profile=private
netsh advfirewall firewall add rule name="Strata pool web app (TCP 8080)" dir=in action=allow protocol=TCP localport=8080 profile=private
echo.
echo Done. Check that this network is set to "Private" in Windows' network settings
echo (Settings, Network and internet, your connection, Network profile type).
pause
