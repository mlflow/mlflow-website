import styles from "./styles.module.css";
import { Blog as BlogType } from "../../posts";
import clsx from "clsx";
import Link from "@docusaurus/Link";

function Blog({ blog }: { blog: BlogType }): JSX.Element {
  const { title, path, tags, authors, date, thumbnail } = blog;
  const author = authors[0];
  return (
    <div className={clsx("card", styles.card)}>
      <div className={clsx("card__body", styles.cardBody)}>
        <Link
          to={path}
          style={{
            textDecoration: "none",
            color: "inherit",
          }}
        >
          <div className={styles.container}>
            <div className={styles.thumbnail}>
              <img
                src={thumbnail || "img/media.png"}
                alt={title}
                title={title}
                className={styles.img}
              />
            </div>
            <div className={styles.body}>
              <div className={styles.tags}>
                {tags.map((tag) => (
                  <a
                    className={clsx(
                      "button button--sm button--outline button--primary",
                      styles.tag,
                    )}
                    href={`/blog/tags/${tag}`}
                  >
                    {tag}
                  </a>
                ))}
              </div>
              <a href={path} className={styles.a}>
                <h4>{title}</h4>
              </a>
              <div className="avatar">
                <a
                  className="avatar__photo-link avatar__photo avatar__photo--md"
                  href={author.url}
                >
                  <img alt={author.name} src={author.image_url} />
                </a>
                <div className="avatar__intro">
                  <small className="avatar__subtitle">
                    by <strong>{author.name}</strong> on{" "}
                    <strong>
                      {new Date(date).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })}
                    </strong>
                  </small>
                </div>
              </div>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}
export default Blog;
