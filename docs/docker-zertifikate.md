# Docker-Downloads hinter HTTPS-Prüfung

Fehler wie `invalid peer certificate: UnknownIssuer` beim Build oder
`x509: certificate signed by unknown authority` bei Ollama können auftreten,
wenn Antivirus-Software (z. B. Norton Web/Mail Shield) oder ein Firmenproxy
HTTPS-Verbindungen prüft. Windows vertraut dessen Stammzertifikat bereits;
Linux-Container haben einen eigenen Zertifikatsspeicher.

Das öffentliche, bereits vertrauenswürdige Stammzertifikat als PEM-Datei unter
`.certs/custom-ca.crt` ablegen. Keine privaten Schlüssel exportieren.
Die Datei bleibt lokal und wird von Git ignoriert. Ohne diese Datei bleibt die
zusätzliche Zertifikatseinbindung inaktiv.

Beispiel für ein bereits im Windows-Computerspeicher vorhandenes Norton-Zertifikat:

```powershell
$certs = @(Get-ChildItem Cert:\LocalMachine\Root | Where-Object {
    $_.Subject -like 'CN=Norton Web/Mail Shield Root,*' -and $_.NotAfter -gt (Get-Date)
})
if ($certs.Count -ne 1) { throw 'Norton-Stammzertifikat nicht eindeutig gefunden.' }
$certs[0] | Format-List Subject, Thumbprint, NotAfter
$pem = "-----BEGIN CERTIFICATE-----`n" +
    [Convert]::ToBase64String($certs[0].RawData, [Base64FormattingOptions]::InsertLineBreaks) +
    "`n-----END CERTIFICATE-----`n"
New-Item -ItemType Directory -Force .certs | Out-Null
[IO.File]::WriteAllText((Join-Path (Get-Location) '.certs/custom-ca.crt'), $pem, [Text.Encoding]::ASCII)
.\setup.ps1 build
```

`setup.ps1` übergibt das Zertifikat als BuildKit-Secret an Backend und Frontend.
Das Backend ergänzt damit seinen CA-Speicher für Paket- und spätere
Modell-Downloads; npm nutzt es zusätzlich während der Paketinstallation.
Ollama erhält die Datei über einen schreibgeschützten Verzeichnismount und
aktualisiert seinen CA-Speicher vor dem Start. Die HTTPS-Prüfung bleibt aktiv.

Manuelle Docker-Builds (auch mit `Dockerfile.gpu-blackwell`) benötigen zusätzlich
`--secret id=custom_ca,src=.certs/custom-ca.crt`. Nach Austausch oder Entfernung
des Zertifikats mit `$env:PROTOKOLL_BUILD_NO_CACHE='true'; .\setup.ps1 build`
neu bauen: Änderungen an Secret-Inhalten invalidieren den Docker-Buildcache
nicht automatisch. Lokal damit gebaute Backend-Images enthalten das zusätzliche
öffentliche Stammzertifikat.

Bei einem anderen Proxy das zugehörige Stammzertifikat aus dem vertrauenswürdigen
Speicher oder von der zuständigen IT beziehen und ebenfalls als PEM ablegen.
