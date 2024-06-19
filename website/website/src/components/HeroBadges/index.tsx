import styles from "./styles.module.css";
import GetStarted from "../GetStarted";
import LatestRelease from "../LatestRelease";

const HeroBadges = () => (
  <div className={styles.container}>
    <GetStarted text="Get Started" />
    <LatestRelease />
  </div>
);

export default HeroBadges;
