; The FLOAT installer: per user, no administrator rights. build.py compiles it with
; /DStageDir=<the staged files> /DOutputDir=<where the .exe goes> /DAppVersion=<version> /DMaxRootLen=<characters>.
; The files are copied here; the libraries are fetched right after by setup\setup_env.py (its own window, or text when
; the installer runs silently). The uninstaller removes only this program's own folders: the user's projects and
; settings live elsewhere and are never touched.

#ifndef StageDir
  #define StageDir "stage"
#endif
#ifndef OutputDir
  #define OutputDir "output"
#endif
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef MaxRootLen
  #define MaxRootLen "79"
#endif

[Setup]
AppId={{C2C8B4E6-6E20-4A6D-9B44-7F3D2A5E9B11}
AppName=FLOAT
AppVersion={#AppVersion}
AppPublisher=FLOAT
DefaultDirName={autopf}\FLOAT
DefaultGroupName=FLOAT
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=FLOAT-Setup
SetupIconFile={#StageDir}\float.ico
UninstallDisplayIcon={app}\float.ico
UninstallDisplayName=FLOAT
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ShowLanguageDialog=no
LanguageDetectionMethod=uilanguage
UsePreviousAppDir=yes
CloseApplications=yes
RestartApplications=no
; the CPU case; the NVIDIA case is checked with the exact figures before anything is downloaded
ExtraDiskSpaceRequired=3221225472

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"

[CustomMessages]
en.DesktopIcon=Create a desktop shortcut
fr.DesktopIcon=Créer un raccourci sur le Bureau
en.LaunchApp=Start FLOAT
fr.LaunchApp=Lancer FLOAT
en.EnvFailed=The libraries could not be installed completely.%n%nFLOAT will offer to finish the installation the next time it is opened from the Start menu.
fr.EnvFailed=Les bibliothèques n'ont pas pu être installées entièrement.%n%nFLOAT proposera de terminer l'installation à sa prochaine ouverture depuis le menu Démarrer.
en.TooLong=This folder's path is too long for Windows.%n%nPlease choose a shorter folder, at most %1 characters.
fr.TooLong=Le chemin de ce dossier est trop long pour Windows.%n%nChoisissez un dossier plus court, de %1 caractères au maximum.

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"; Flags: unchecked

[InstallDelete]
; old code of a former version
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\setup"
Type: filesandordirs; Name: "{app}\python"

[Files]
Source: "{#StageDir}\*"; Excludes: "float.iss"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\FLOAT"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\setup\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\float.ico"
Name: "{autodesktop}\FLOAT"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\setup\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\float.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\setup\launch.py"""; WorkingDir: "{app}"; Description: "{cm:LaunchApp}"; Flags: postinstall nowait skipifsilent; Check: EnvReady

[UninstallDelete]
; what the program made beside the files it copied, by name: never the install folder as a whole, where a user may
; have put something of his own, and nothing outside it (projects and settings stay)
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\setup"
Type: filesandordirs; Name: "{app}\downloads"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\logs"
Type: files; Name: "{app}\setup.log"

[Code]
var
  EnvOk: Boolean;

function EnvReady: Boolean;
begin
  Result := EnvOk;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  { every installed file must stay under Windows' 260 characters: neither Setup nor the uninstaller handles longer ones }
  { silent: refusing the page would hang; setup_env.py refuses the folder instead }
  if (CurPageID = wpSelectDir) and (not WizardSilent) and (Length(WizardDirValue) > {#MaxRootLen}) then
  begin
    MsgBox(FmtMessage(ExpandConstant('{cm:TooLong}'), ['{#MaxRootLen}']), mbError, MB_OK);
    Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  Py, Args: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  Args := '"' + ExpandConstant('{app}\setup\setup_env.py') + '" --root "' + ExpandConstant('{app}') + '"';
  if WizardSilent then
  begin
    Py := ExpandConstant('{app}\python\python.exe');
    Args := Args + ' --no-gui';
    EnvOk := Exec(Py, Args, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0);
  end
  else
  begin
    Py := ExpandConstant('{app}\python\pythonw.exe');
    WizardForm.Hide;
    EnvOk := Exec(Py, Args, ExpandConstant('{app}'), SW_SHOW, ewWaitUntilTerminated, Code) and (Code = 0);
    WizardForm.Show;
    if not EnvOk then
      MsgBox(ExpandConstant('{cm:EnvFailed}'), mbInformation, MB_OK);
  end;
  Log(Format('library setup: result %d', [Code]));
end;
