# Chassis fan control on motherboard headers (GPU-temp-linked)

MI50s are blowers: they dump every watt into chassis air. On a 4-card rig
that means the *chassis* fans — not the GPU fans — decide sustained
junction temps, and most boards either run them at full scream or ignore GPU
temperature entirely. This is the setup we run: the two 80 mm fans blowing on
the GPU bank are driven off the motherboard's Super I/O chip (Nuvoton
NCT6779 via `nct6775`/hwmon), keyed to the hottest MI50 junction.

Quiet at idle, and the monitor screenshot in the README (47 °C average under
real traffic) is this loop doing its job.

## Files here

- [`gpu-fancontrol.sh`](gpu-fancontrol.sh) — the controller (edit the CONFIG
  block: PWM headers, GPU count, PCI device id)
- [`gpu-fancontrol.service`](gpu-fancontrol.service) — systemd unit
- persist the chip with `/etc/modules-load.d/` containing one line: `nct6775`

## Getting your mapping right (do not skip)

1. `modprobe nct6775 && sensors` — if nothing shows, find your chip with
   `sensors-detect`. Any Nuvoton/IT87xx-family chip works the same way.
2. **Dip-test every header**: `echo 0 > pwmN_enable; echo 60 > pwmN` and
   watch which physical fan slows. Write your mapping down.
3. **Check header current ratings** against your fans. Our two GPU fans are
   0.7 A each and came shipped on ONE header via splitter — 1.4 A on a 1 A
   header. We split them across two headers; do the math on yours.
4. Resolve hwmon dirs **by chip name, never `hwmonNN`** — the index shifts
   between boots and even mid-session when the chip re-registers (the script
   re-resolves every cycle).

## The pitfall that defines the design

Firmware holds the PWMs in smart-fan mode (`pwmN_enable=5`) and **reasserts
it periodically**: writes fail with `EBUSY` until you write
`echo 0 > pwmN_enable`, and even after that the firmware steals control back
on its own schedule. So this is not a set-and-forget sysfs write — the
controller re-claims manual mode every 5 s cycle. Any one-shot fancontrol
setup on these boards is a race you lose eventually.

## Curve

Keys on the hottest junction across all cards: ≤40 °C floor (~35 %),
60 °C → 55 %, 75 °C → 78 %, ≥85 °C → full. Ramps up immediately, eases down
(max −15/cycle, ±6 deadband) so fan noise doesn't breathe.

## Honest thermal reality check

A 72K-token prefill drove junctions to 96 °C **with chassis fans at 100 %**
— chassis airflow is not the bottleneck on an MI50 bank, the blowers
themselves are. This controller buys silence at light load and sanity at
heavy load; it does not defeat physics, and huge prefills will spike toward
100 °C regardless. (The platinum engine's chunked prefill keeps normal
traffic far from that regime.)
