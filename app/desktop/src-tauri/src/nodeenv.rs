//! Keep an explicit Node CA choice without inheriting a contradictory default.
use std::ffi::OsStr;
use std::process::Command;

pub fn configure(command: &mut Command) {
    apply_options(command, std::env::var_os("NODE_OPTIONS").as_deref());
}

fn apply_options(command: &mut Command, options: Option<&OsStr>) {
    let openssl = options.map(|value| value.to_string_lossy().split_ascii_whitespace().any(|part| {
        matches!(part.trim_matches('"'), "--use-openssl-ca" | "--use_openssl_ca")
    })).unwrap_or(false);
    if openssl {
        // Node 24 rejects this combination before loading the host. The explicit
        // NODE_OPTIONS choice takes precedence; extra certificates and all other
        // caller options remain intact. Only this child's environment changes.
        command.env_remove("NODE_USE_SYSTEM_CA");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn policy(options: Option<&str>) -> Command {
        let mut command = Command::new("node");
        command.env("NODE_USE_SYSTEM_CA", "1").env("NODE_EXTRA_CA_CERTS", "synthetic.pem");
        if let Some(value) = options { command.env("NODE_OPTIONS", value); }
        apply_options(&mut command, options.map(OsStr::new));
        command
    }
    fn value(command: &Command, key: &str) -> Option<Option<String>> {
        command.get_envs().find(|(name, _)| *name == OsStr::new(key)).map(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }
    #[test]
    fn explicit_openssl_choice_drops_only_the_conflicting_default() {
        let command = policy(Some("--max-old-space-size=2048 --use-openssl-ca"));
        assert_eq!(value(&command, "NODE_USE_SYSTEM_CA"), Some(None));
        assert_eq!(value(&command, "NODE_OPTIONS"), Some(Some("--max-old-space-size=2048 --use-openssl-ca".into())));
        assert_eq!(value(&command, "NODE_EXTRA_CA_CERTS"), Some(Some("synthetic.pem".into())));
    }
    #[test]
    fn quoted_and_underscore_options_are_recognised() {
        for options in ["\"--use-openssl-ca\"", "--use_openssl_ca", "--require \"C:/path with spaces/loader.js\" --use-openssl-ca"] {
            assert_eq!(value(&policy(Some(options)), "NODE_USE_SYSTEM_CA"), Some(None));
        }
    }
    #[test]
    fn ordinary_options_keep_system_ca() {
        assert_eq!(value(&policy(Some("--max-old-space-size=2048")), "NODE_USE_SYSTEM_CA"), Some(Some("1".into())));
    }
    #[test]
    fn absent_options_keep_system_ca() {
        assert_eq!(value(&policy(None), "NODE_USE_SYSTEM_CA"), Some(Some("1".into())));
    }
    #[test]
    fn a_similar_filename_is_not_a_ca_switch() {
        assert_eq!(value(&policy(Some("--require C:/--use-openssl-ca.js")), "NODE_USE_SYSTEM_CA"), Some(Some("1".into())));
    }
}
